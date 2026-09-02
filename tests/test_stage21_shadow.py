from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import write_canonical_history_csv
from backend.app.research.prospective import (
    DIAGNOSIS_INSUFFICIENT_DATA,
    ProspectiveRecord,
    RandomControl,
    save_prospective_record,
)
from backend.app.research.shadow import (
    SHADOW_EVALUATED,
    SHADOW_PENDING,
    STATUS_ACTIVE_SHADOW,
    TEST_MODEL_NAME,
    evaluate_shadow_predictions,
    generate_shadow_prediction,
    load_shadow_record,
    register_shadow_challenger,
    save_shadow_record,
    shadow_record_path,
    shadow_summary,
)


def _draws(
    lottery: LotteryDefinition,
    *,
    count: int = 112,
    start_number: int = 1000,
    start_date: date = date(2024, 1, 1),
) -> tuple[HistoricalDraw, ...]:
    step = 3 if str(lottery.code) == "LOTO6" else 7
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % (lottery.number_max - lottery.numbers_per_ticket)) + 1
        main = tuple(
            sorted(
                ((start + offset * stride - 1) % lottery.number_max) + 1
                for offset in range(lottery.numbers_per_ticket)
            )
        )
        bonus = next(
            number
            for number in range(lottery.number_min, lottery.number_max + 1)
            if number not in main
        )
        draws.append(
            HistoricalDraw(
                lottery,
                start_number + index,
                start_date + timedelta(days=index * step),
                main,
                (bonus,),
            )
        )
    return tuple(draws)


def _register(root: Path, lottery: LotteryDefinition = LOTO6, *, challenger_id: str = "shadow-a"):
    return register_shadow_challenger(
        challenger_id=challenger_id,
        lottery=lottery,
        model=TEST_MODEL_NAME,
        feature_group="gap_only" if str(lottery.code) == "LOTO6" else "pair_only",
        status=STATUS_ACTIVE_SHADOW,
        seed=123456,
        minimum_evaluation_draws=5,
        root=root,
        registered_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _champion_record(
    lottery: LotteryDefinition,
    draw: HistoricalDraw,
    root: Path,
) -> None:
    record = ProspectiveRecord(
        schema_version="stage19-prospective-record-v1",
        lottery=str(lottery.code),
        draw_number=draw.draw_number,
        draw_date=draw.draw_date.isoformat(),
        prediction_id=str(draw.draw_number),
        prediction_path="data/predictions/test.json",
        generated_at="2026-01-01T00:00:00+00:00",
        evaluated_at="2026-01-02T00:00:00+00:00",
        dataset_hash="dataset",
        model="champion",
        feature_group="champion",
        feature_version="v",
        feature_names=(),
        portfolio_method="top_ranked",
        seed=123456,
        tickets_per_draw=3,
        actual_result={
            "main_numbers": draw.main_numbers,
            "bonus_numbers": draw.bonus_numbers,
        },
        ticket_results=(),
        portfolio={
            "best_ticket_matches": 1,
            "total_main_matches_across_tickets": 2,
            "prize_qualified_tickets": 0,
            "paper_cost": 600,
            "paper_gross": None,
            "paper_net": None,
        },
        ranking={"top_capture": {"top_6": 1}, "winning_number_ranks": (), "top_numbers": ()},
        random_control=RandomControl(
            replications=10,
            seed=123456,
            tickets_per_draw=3,
            mean_best_ticket_matches=1.0,
            mean_total_portfolio_matches=2.0,
            prize_qualified_rate=0.0,
            production_best_match_percentile=0.5,
            production_total_match_percentile=0.5,
            paper_gross_distribution=None,
        ),
        diagnostic_classification=DIAGNOSIS_INSUFFICIENT_DATA,
        warnings=(),
    )
    save_prospective_record(record, root / str(lottery.code) / f"{draw.draw_number}.json")


def test_register_shadow_challenger_and_duplicate_rejected(tmp_path: Path) -> None:
    challenger = _register(tmp_path)

    assert challenger.status == STATUS_ACTIVE_SHADOW
    assert challenger.model == TEST_MODEL_NAME
    with pytest.raises(ResearchValidationError, match="duplicate shadow challenger"):
        _register(tmp_path)


def test_retired_experiment_cannot_activate_without_override(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError, match="retired experiment"):
        register_shadow_challenger(
            challenger_id="retired-exp",
            lottery=LOTO6,
            model=TEST_MODEL_NAME,
            feature_group="gap_only",
            status=STATUS_ACTIVE_SHADOW,
            research_experiment_status="RETIRE",
            root=tmp_path,
        )


def test_generate_shadow_prediction_is_separate_immutable_and_eligible(tmp_path: Path) -> None:
    _register(tmp_path)
    training = _draws(LOTO6)[:-1]

    result = generate_shadow_prediction(
        training,
        LOTO6,
        "shadow-a",
        tickets_per_draw=3,
        root=tmp_path,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    duplicate = generate_shadow_prediction(
        training, LOTO6, "shadow-a", tickets_per_draw=3, root=tmp_path
    )

    assert result.record.status == SHADOW_PENDING
    assert result.record.prospective_eligible is True
    assert result.record.ticket_count == 3
    assert len({ticket.numbers for ticket in result.record.tickets}) == 3
    assert "data/predictions" not in result.record_path.replace("\\", "/")
    assert duplicate.existing_record is True
    with pytest.raises(ResearchValidationError, match="conflicting shadow record"):
        save_shadow_record(
            replace(result.record, seed=999),
            shadow_record_path(tmp_path, LOTO6, "shadow-a", result.record.target_draw_number),
        )


def test_generated_after_draw_is_rejected(tmp_path: Path) -> None:
    _register(tmp_path)
    training = _draws(LOTO6)[:-1]
    target_date = training[-1].draw_date + timedelta(days=3)

    with pytest.raises(ResearchValidationError, match="generated after target"):
        generate_shadow_prediction(
            training,
            LOTO6,
            "shadow-a",
            root=tmp_path,
            generated_at=datetime.combine(target_date, datetime.max.time(), tzinfo=UTC),
        )


def test_evaluate_shadow_prediction_with_champion_and_random_comparison(tmp_path: Path) -> None:
    _register(tmp_path)
    history = _draws(LOTO6)
    generated = generate_shadow_prediction(
        history[:-1],
        LOTO6,
        "shadow-a",
        tickets_per_draw=3,
        root=tmp_path,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    prospective_root = tmp_path / "prospective"
    _champion_record(LOTO6, history[-1], prospective_root)

    result = evaluate_shadow_predictions(
        history,
        LOTO6,
        challenger_id="shadow-a",
        root=tmp_path,
        prospective_root=prospective_root,
        random_replications=20,
    )
    record = load_shadow_record(generated.record_path)

    assert result["evaluated_count"] == 1
    assert record.status == SHADOW_EVALUATED
    assert record.champion_comparison is not None
    assert record.random_control is not None
    assert record.random_control["tickets_per_draw"] == 3


def test_shadow_summary_reports_insufficient_data(tmp_path: Path) -> None:
    _register(tmp_path, MINI_LOTO)
    history = _draws(MINI_LOTO)
    generate_shadow_prediction(
        history[:-1],
        MINI_LOTO,
        "shadow-a",
        tickets_per_draw=2,
        root=tmp_path,
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    evaluate_shadow_predictions(
        history,
        MINI_LOTO,
        challenger_id="shadow-a",
        root=tmp_path,
        random_replications=10,
    )

    summary = shadow_summary(lottery=MINI_LOTO, root=tmp_path)

    assert summary["challengers"][0]["evaluated_draws"] == 1
    assert summary["challengers"][0]["conclusion"] == DIAGNOSIS_INSUFFICIENT_DATA
    assert summary["production_safety"]["auto_promotion"] is False


def test_shadow_does_not_mutate_production_settlement_or_notification_files(tmp_path: Path) -> None:
    production = tmp_path / "predictions" / "LOTO6" / "2132.json"
    settlement = tmp_path / "settlements" / "LOTO6" / "2131.json"
    notification = tmp_path / "notifications" / "notice.json"
    for path in (production, settlement, notification):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"immutable": true}', encoding="utf-8")
    before = {path: path.read_bytes() for path in (production, settlement, notification)}

    _register(tmp_path / "shadow")
    history = _draws(LOTO6)
    generate_shadow_prediction(
        history[:-1],
        LOTO6,
        "shadow-a",
        root=tmp_path / "shadow",
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    evaluate_shadow_predictions(history, LOTO6, root=tmp_path / "shadow", random_replications=5)

    assert {path: path.read_bytes() for path in before} == before


def test_stage21_cli_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "loto6.csv"
    shadow_root = tmp_path / "shadow"
    write_canonical_history_csv(_draws(LOTO6, start_date=date(2026, 1, 1))[:-1], data_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--shadow-root",
            str(shadow_root),
            "--challenger-id",
            "cli-shadow",
            "--model",
            TEST_MODEL_NAME,
            "--feature-group",
            "gap_only",
            "register-shadow-challenger",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--data",
            str(data_path),
            "--shadow-root",
            str(shadow_root),
            "--challenger-id",
            "cli-shadow",
            "--tickets-per-draw",
            "2",
            "generate-shadow-prediction",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["challenger_id"] == "cli-shadow"
    assert payload["status"] == SHADOW_PENDING
