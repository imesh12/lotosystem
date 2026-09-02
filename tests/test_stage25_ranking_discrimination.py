from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.stage25_ranking_discrimination import (
    STAGE25_CHAMPION_C,
    STAGE25_DISCOVERY_CUTOFF_DRAW,
    STAGE25_HOLDOUT_DRAW,
    champion_configuration,
    discovery_slice,
    monotonic_transform_rankings,
    run_stage25_leakage_audit,
    run_stage25_ranking_discrimination,
    stable_payload_hash,
)


def _draws(
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    start_number: int = 1370,
    count: int = 34,
    start_date: date = date(2026, 1, 20),
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for index in range(count):
        base = ((index * 11) % lottery.number_max) + 1
        main = tuple(
            sorted(
                ((base + offset * 6 - 1) % lottery.number_max) + 1
                for offset in range(lottery.numbers_per_ticket)
            )
        )
        bonus = next(
            number
            for number in range(lottery.number_min, lottery.number_max + 1)
            if number not in main
        )
        rows.append(
            HistoricalDraw(
                lottery=lottery,
                draw_number=start_number + index,
                draw_date=start_date + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(rows)


def _run(draws: tuple[HistoricalDraw, ...], tmp_path: Path | None = None):
    return run_stage25_ranking_discrimination(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        refit_interval=5,
        bootstrap_replications=25,
        output_dir=tmp_path / "stage25" if tmp_path else None,
    )


def test_stage25_is_mini_loto_only() -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        run_stage25_ranking_discrimination(
            _draws(LOTO6, start_number=2100, count=34),
            LOTO6,
            min_training_draws=10,
            bootstrap_replications=10,
        )


def test_discovery_cutoff_excludes_1402_and_later_draws() -> None:
    discovery = discovery_slice(_draws(), MINI_LOTO)

    assert discovery[-1].draw_number == STAGE25_DISCOVERY_CUTOFF_DRAW
    assert all(draw.draw_number <= STAGE25_DISCOVERY_CUTOFF_DRAW for draw in discovery)
    assert STAGE25_HOLDOUT_DRAW not in {draw.draw_number for draw in discovery}
    assert 1403 not in {draw.draw_number for draw in discovery}


def test_stage25_result_is_reproducible_and_writes_frozen_decision(tmp_path: Path) -> None:
    first = _run(_draws(), tmp_path)
    second = _run(_draws())
    decision_path = tmp_path / "stage25" / "v2_stage25_frozen_decision.json"

    assert decision_path.exists()
    assert first.discovery_dataset_hash == second.discovery_dataset_hash
    assert first.frozen_decision_hash == second.frozen_decision_hash
    assert first.frozen_decision_hash == stable_payload_hash(
        {key: value for key, value in first.frozen_decision.items() if key != "decision_hash"}
    )


def test_1402_holdout_cannot_change_frozen_decision() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number == STAGE25_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = _run(draws)
    changed_holdout = _run(mutated)

    assert original.discovery_dataset_hash == changed_holdout.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_holdout.frozen_decision_hash
    assert original.holdout.actual_main_numbers != changed_holdout.holdout.actual_main_numbers


def test_later_draws_cannot_change_decision_or_holdout() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number > STAGE25_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = _run(draws)
    changed_future = _run(mutated)

    assert original.discovery_dataset_hash == changed_future.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_future.frozen_decision_hash
    assert original.holdout == changed_future.holdout


def test_monotonic_score_transforms_do_not_change_rankings() -> None:
    scores = {1: 0.40, 2: 0.55, 3: 0.50, 4: 0.52, 5: 0.48}

    rankings = monotonic_transform_rankings(scores)

    assert rankings["probability"] == rankings["decision_function_equivalent"]
    assert rankings["probability"] == rankings["z_score"]
    assert rankings["probability"] == rankings["percentile"]


def test_stage25_leakage_audit_passes_for_target_and_future_mutations() -> None:
    audit = run_stage25_leakage_audit(
        discovery_slice(_draws(), MINI_LOTO),
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
    )

    assert audit.lookahead_safe is True
    assert audit.training_dates_strictly_before_target is True
    assert audit.target_mutation_changes_features is False
    assert audit.future_mutation_changes_prediction is False


def test_calibration_and_regularization_results_are_present() -> None:
    result = _run(_draws())

    assert set(result.calibration_results) == {"uncalibrated", "sigmoid", "isotonic"}
    assert "c_0_01_uncalibrated" in result.regularization_results
    assert "c_10_0_uncalibrated" in result.regularization_results
    assert "c_1_0_sigmoid" in result.regularization_results
    assert set(result.regularization_results["c_1_0_uncalibrated"].primary_comparisons) == {
        "mean_winner_rank",
        "top15_capture_rate",
        "top5_capture_rate",
    }


def test_champion_configuration_is_preserved() -> None:
    config = champion_configuration(seed=123456)

    assert config["model"] == "logistic_regression"
    assert config["feature_group"] == "pair_only"
    assert config["feature_names"] == ("pair_strength_rate",)
    assert config["portfolio_method"] == "top_ranked"
    assert config["calibration"] == "uncalibrated"
    assert config["model_parameters"]["C"] == STAGE25_CHAMPION_C
    assert config["model_parameters"]["solver"] == "liblinear"
    assert config["model_parameters"]["class_weight"] == "balanced"


def test_stage25_does_not_mutate_runtime_paths(tmp_path: Path) -> None:
    watched = (
        Path("data/processed/mini_loto_history.csv"),
        Path("data/predictions/MINI_LOTO/1402.json"),
        Path("data/settlements/ledger.json"),
        Path("config/operational_settings.json"),
    )
    before = {path: path.read_bytes() for path in watched if path.exists()}

    _run(_draws(), tmp_path)

    after = {path: path.read_bytes() for path in watched if path.exists()}
    assert before == after
