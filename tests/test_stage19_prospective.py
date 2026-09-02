from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import write_canonical_history_csv
from backend.app.research.persistence import research_result_json
from backend.app.research.production import (
    PREDICTION_STATUS_EVALUATED,
    evaluate_pending_predictions,
    generate_next_prediction,
    load_prediction_record,
)
from backend.app.research.prospective import (
    DIAGNOSIS_INSUFFICIENT_DATA,
    evaluate_prediction_record,
    prospective_evaluate,
    prospective_record_path,
    prospective_summary,
    save_prospective_record,
)
from backend.app.research.settlement import build_settlement, save_settlement, settlement_path


def _draws(
    lottery: LotteryDefinition,
    *,
    start_number: int,
    count: int,
    start_date: date,
) -> tuple[HistoricalDraw, ...]:
    step = 3 if str(lottery.code) == "LOTO6" else 7
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
    draws = []
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


def _evaluated_prediction_with_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lottery: LotteryDefinition,
) -> Path:
    history = _draws(lottery, start_number=1000, count=112, start_date=date(2024, 1, 1))
    training = history[:-1]
    full_history_path = tmp_path / f"{lottery.code}_history.csv"
    write_canonical_history_csv(history, full_history_path)
    monkeypatch.setattr(
        "backend.app.research.prospective.canonical_history_path",
        lambda _: full_history_path,
    )
    prediction_root = tmp_path / "predictions"
    settlement_root = tmp_path / "settlements"
    generated_at = datetime.combine(
        training[-1].draw_date,
        datetime.min.time(),
        tzinfo=UTC,
    )
    result = generate_next_prediction(
        training,
        lottery,
        ResearchConfig(seed=123456),
        tickets_per_draw=3,
        prediction_root=prediction_root,
        generated_at=generated_at,
    )
    evaluate_pending_predictions(history, lottery, prediction_root=prediction_root)
    record = load_prediction_record(result.record_path)
    settlement = build_settlement(record, lottery, result.record_path, ())
    save_settlement(
        settlement,
        settlement_path(settlement_root, lottery, record.target_draw_number),
    )
    return Path(result.record_path)


def test_eligible_loto6_prediction_creates_ranking_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)

    record = evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=10,
    )

    assert record.lottery == "LOTO6"
    assert len(record.ranking["winning_number_ranks"]) == 6
    assert set(record.ranking["top_capture"]) == {"top_6", "top_12", "top_18", "top_24"}
    assert record.random_control.replications == 10
    assert record.random_control.tickets_per_draw == 3
    assert prospective_record_path(tmp_path / "prospective", LOTO6, record.draw_number).exists()


def test_mini_loto_ranking_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, MINI_LOTO)

    record = evaluate_prediction_record(
        prediction_path,
        MINI_LOTO,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=10,
    )

    assert len(record.ranking["winning_number_ranks"]) == 5
    assert set(record.ranking["top_capture"]) == {"top_5", "top_10", "top_15", "top_20"}


def test_prediction_generated_after_draw_is_not_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)
    record = load_prediction_record(prediction_path)
    late = replace(record, generated_at=f"{record.target_draw_date}T23:00:00+09:00")
    prediction_path.write_text(research_result_json(late), encoding="utf-8")

    with pytest.raises(ResearchValidationError, match="NOT_ELIGIBLE"):
        evaluate_prediction_record(
            prediction_path,
            LOTO6,
            settlement_root=tmp_path / "settlements",
            prospective_root=tmp_path / "prospective",
            random_replications=5,
        )


def test_missing_prediction_is_reported_not_eligible(
    tmp_path: Path,
) -> None:
    result = prospective_evaluate(
        lottery=LOTO6,
        prediction_root=tmp_path / "missing",
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=5,
    )

    assert result["eligible_records"] == 0
    assert result["summary"]["lotteries"]["LOTO6"]["eligible_draws"] == 0


def test_random_control_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)

    first = evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=20,
    )
    second = evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=20,
    )

    assert first.random_control == second.random_control
    assert first.ranking == second.ranking


def test_summary_one_draw_is_insufficient_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)
    evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=10,
    )

    summary = prospective_summary(lottery=LOTO6, prospective_root=tmp_path / "prospective")

    assert summary["lotteries"]["LOTO6"]["eligible_draws"] == 1
    assert summary["lotteries"]["LOTO6"]["conclusion"] == DIAGNOSIS_INSUFFICIENT_DATA


def test_idempotent_persistence_and_conflict_detection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)
    record = evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=10,
    )
    path = prospective_record_path(tmp_path / "prospective", LOTO6, record.draw_number)

    assert save_prospective_record(record, path) == path
    conflicting = replace(record, diagnostic_classification="RANKING_WEAK")
    if conflicting == record:
        conflicting = replace(record, diagnostic_classification="PORTFOLIO_WEAK")
    with pytest.raises(ResearchValidationError, match="conflicting prospective record"):
        save_prospective_record(conflicting, path)


def test_prospective_evaluation_does_not_mutate_production_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prediction_path = _evaluated_prediction_with_settlement(monkeypatch, tmp_path, LOTO6)
    settlement_file = next((tmp_path / "settlements" / "LOTO6").glob("*.json"))
    prediction_before = prediction_path.read_bytes()
    settlement_before = settlement_file.read_bytes()

    evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root=tmp_path / "settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=10,
    )

    assert prediction_path.read_bytes() == prediction_before
    assert settlement_file.read_bytes() == settlement_before


def test_real_loto6_2131_smoke_if_local_data_available(tmp_path: Path) -> None:
    prediction_path = Path("data/predictions/LOTO6/2131.json")
    settlement_file = Path("data/settlements/LOTO6/2131.json")
    history_path = Path("data/processed/loto6_history.csv")
    if not (prediction_path.exists() and settlement_file.exists() and history_path.exists()):
        return

    record = evaluate_prediction_record(
        prediction_path,
        LOTO6,
        settlement_root="data/settlements",
        prospective_root=tmp_path / "prospective",
        random_replications=5,
    )
    source_prediction = load_prediction_record(prediction_path)
    history = load_draws_csv(history_path, LOTO6)

    assert source_prediction.status == PREDICTION_STATUS_EVALUATED
    assert history[-1].draw_number >= 2131
    assert record.draw_number == 2131
    assert tuple(record.actual_result["main_numbers"]) == (25, 26, 28, 33, 35, 43)
    assert len(record.ranking["winning_number_ranks"]) == 6
    assert record.portfolio["paper_net"] == -600
