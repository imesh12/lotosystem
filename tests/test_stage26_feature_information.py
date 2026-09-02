from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.stage26_feature_information import (
    FEATURE_DIRECTIONS,
    STAGE26_DISCOVERY_CUTOFF_DRAW,
    STAGE26_HOLDOUT_DRAW,
    discovery_slice,
    feature_inventory,
    run_stage26_feature_information_audit,
    run_stage26_leakage_audit,
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
    return run_stage26_feature_information_audit(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        output_dir=tmp_path / "stage26" if tmp_path else None,
    )


def test_stage26_is_mini_loto_only() -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        run_stage26_feature_information_audit(
            _draws(LOTO6, start_number=2100, count=34),
            LOTO6,
            min_training_draws=10,
            bootstrap_replications=10,
        )


def test_discovery_cutoff_excludes_1402_and_later_draws() -> None:
    discovery = discovery_slice(_draws(), MINI_LOTO)

    assert discovery[-1].draw_number == STAGE26_DISCOVERY_CUTOFF_DRAW
    assert STAGE26_HOLDOUT_DRAW not in {draw.draw_number for draw in discovery}
    assert 1403 not in {draw.draw_number for draw in discovery}


def test_feature_inventory_is_deterministic_and_historical_only() -> None:
    first = feature_inventory()
    second = feature_inventory()

    assert first == second
    assert first["pair_strength_rate"].expected_direction == "higher"
    assert first["current_gap"].expected_direction == "lower"
    assert all(not item.target_draw_included for item in first.values())
    assert set(first) == set(FEATURE_DIRECTIONS)


def test_stage26_reproducibility_and_frozen_decision_hash(tmp_path: Path) -> None:
    first = _run(_draws(), tmp_path)
    second = _run(_draws())

    assert (tmp_path / "stage26" / "v2_stage26_frozen_decision.json").exists()
    assert first.discovery_dataset_hash == second.discovery_dataset_hash
    assert first.frozen_decision_hash == second.frozen_decision_hash
    assert first.frozen_decision_hash == stable_payload_hash(
        {key: value for key, value in first.frozen_decision.items() if key != "decision_hash"}
    )


def test_1402_holdout_cannot_change_frozen_decision() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number == STAGE26_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = _run(draws)
    changed_holdout = _run(mutated)

    assert original.discovery_dataset_hash == changed_holdout.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_holdout.frozen_decision_hash
    assert original.holdout.actual_main_numbers != changed_holdout.holdout.actual_main_numbers


def test_later_draws_cannot_change_discovery_or_holdout() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number > STAGE26_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = _run(draws)
    changed_future = _run(mutated)

    assert original.discovery_dataset_hash == changed_future.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_future.frozen_decision_hash
    assert original.holdout == changed_future.holdout


def test_target_and_future_mutations_do_not_change_prior_features() -> None:
    audit = run_stage26_leakage_audit(
        discovery_slice(_draws(), MINI_LOTO),
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
    )

    assert audit.lookahead_safe is True
    assert audit.training_dates_strictly_before_target is True
    assert audit.target_mutation_changes_features is False
    assert audit.future_mutation_changes_prediction is False


def test_direct_pair_strength_and_lr_attribution_are_deterministic() -> None:
    first = _run(_draws())
    second = _run(_draws())

    assert first.features["pair_strength_rate"].direct_ranker == (
        second.features["pair_strength_rate"].direct_ranker
    )
    assert first.champion_attribution == second.champion_attribution
    assert first.champion_attribution.direct_feature == "pair_strength_rate"
    assert first.champion_attribution.lr_feature_group == "pair_only"


def test_time_segments_rolling_correlations_and_inverse_diagnostic_exist() -> None:
    result = _run(_draws())
    pair = result.features["pair_strength_rate"]

    assert pair.period_stability
    assert pair.rolling_stability.window_size == 100
    assert pair.rolling_stability.window_count == 0
    assert result.redundancy.average_pairwise_spearman >= 0
    assert result.redundancy.pair_strength_redundancy
    assert pair.inverse_is_diagnostic is True
    assert pair.inverse_ranker.mean_winner_rank > 0


def test_holm_bh_and_recommendations_are_deterministic() -> None:
    result = _run(_draws())
    primary = result.features["pair_strength_rate"].direct_ranker.primary_endpoints

    assert set(primary) == {"mean_winner_rank", "top5_capture_rate", "top15_capture_rate"}
    assert all(item.holm_p_value >= item.raw_p_value for item in primary.values())
    assert all(item.bh_p_value >= item.raw_p_value for item in primary.values())
    assert result.stage27_feature_recommendation in {"NONE", *FEATURE_DIRECTIONS}
    assert result.stage27_ensemble_recommendation == "NONE" or "+" in (
        result.stage27_ensemble_recommendation
    )


def test_stage26_does_not_mutate_runtime_paths(tmp_path: Path) -> None:
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
