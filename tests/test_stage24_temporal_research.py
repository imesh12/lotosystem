from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.stage24_temporal_research import (
    STAGE24_DISCOVERY_CUTOFF_DRAW,
    STAGE24_HOLDOUT_DRAW,
    discovery_slice,
    evaluate_holdout_after_frozen_decision,
    run_stage24_temporal_research,
    score_temporal_signal,
    stable_payload_hash,
    strongest_signal,
)


def _draws(
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    start_number: int = 1375,
    count: int = 29,
    start_date: date = date(2026, 2, 24),
) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        base = ((index * 7) % lottery.number_max) + 1
        main = tuple(
            sorted(
                ((base + offset * 5 - 1) % lottery.number_max) + 1
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
                lottery=lottery,
                draw_number=start_number + index,
                draw_date=start_date + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def test_discovery_cutoff_excludes_1402_and_later_draws() -> None:
    draws = _draws()

    discovery = discovery_slice(draws, MINI_LOTO)

    assert discovery[-1].draw_number == STAGE24_DISCOVERY_CUTOFF_DRAW
    assert all(draw.draw_number <= STAGE24_DISCOVERY_CUTOFF_DRAW for draw in discovery)
    assert STAGE24_HOLDOUT_DRAW not in {draw.draw_number for draw in discovery}


def test_loto6_rejected_for_stage24() -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        run_stage24_temporal_research(
            _draws(LOTO6, start_number=2000, count=110),
            LOTO6,
            bootstrap_replications=10,
            monte_carlo_replications=10,
        )


def test_stage24_reproducibility_and_frozen_decision_hash(tmp_path: Path) -> None:
    draws = _draws()

    first = run_stage24_temporal_research(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
        output_dir=tmp_path / "stage24",
    )
    second = run_stage24_temporal_research(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
    )
    decision_path = tmp_path / "stage24" / "v2_stage24_frozen_decision.json"

    assert decision_path.exists()
    assert first.discovery_dataset_hash == second.discovery_dataset_hash
    assert first.frozen_decision_hash == second.frozen_decision_hash
    assert first.frozen_decision_hash == stable_payload_hash(
        {key: value for key, value in first.frozen_decision.items() if key != "decision_hash"}
    )
    assert json.loads(decision_path.read_text(encoding="utf-8"))["decision_hash"] == (
        first.frozen_decision_hash
    )


def test_1402_holdout_cannot_influence_discovery_selection() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number == STAGE24_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = run_stage24_temporal_research(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
    )
    changed_holdout = run_stage24_temporal_research(
        mutated,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
    )

    assert original.discovery_dataset_hash == changed_holdout.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_holdout.frozen_decision_hash
    assert (
        strongest_signal(original.signals).signal
        == strongest_signal(changed_holdout.signals).signal
    )
    assert original.holdout.matches != changed_holdout.holdout.matches


def test_later_draws_cannot_influence_discovery_or_holdout() -> None:
    draws = _draws()
    mutated = tuple(
        replace(draw, main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,))
        if draw.draw_number > STAGE24_HOLDOUT_DRAW
        else draw
        for draw in draws
    )

    original = run_stage24_temporal_research(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
    )
    changed_future = run_stage24_temporal_research(
        mutated,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
    )

    assert original.discovery_dataset_hash == changed_future.discovery_dataset_hash
    assert original.frozen_decision_hash == changed_future.frozen_decision_hash
    assert original.holdout == changed_future.holdout


def test_signal_features_use_only_prior_history() -> None:
    draws = _draws(count=120)
    history = draws[:100]
    target_mutated = (
        *history,
        replace(draws[100], main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,)),
    )
    future_mutated = (
        *history,
        draws[100],
        replace(draws[101], main_numbers=(1, 2, 3, 4, 5), bonus_numbers=(31,)),
    )

    scores = score_temporal_signal("multi_lag_recurrence_t2", history, MINI_LOTO)
    target_scores = score_temporal_signal(
        "multi_lag_recurrence_t2", target_mutated[:100], MINI_LOTO
    )
    future_scores = score_temporal_signal(
        "multi_lag_recurrence_t2", future_mutated[:100], MINI_LOTO
    )

    assert scores == target_scores == future_scores


def test_run_does_not_mutate_production_runtime_paths(tmp_path: Path) -> None:
    draws = _draws()
    watched = (
        Path("data/processed/mini_loto_history.csv"),
        Path("data/predictions/MINI_LOTO/1402.json"),
        Path("data/settlements/ledger.json"),
    )
    before = {path: path.read_bytes() for path in watched if path.exists()}

    run_stage24_temporal_research(
        draws,
        MINI_LOTO,
        seed=123456,
        min_training_draws=10,
        bootstrap_replications=25,
        monte_carlo_replications=25,
        output_dir=tmp_path / "stage24",
    )

    after = {path: path.read_bytes() for path in watched if path.exists()}
    assert before == after


def test_discovery_hash_is_independent_of_post_cutoff_data() -> None:
    draws = _draws()
    discovery = discovery_slice(draws, MINI_LOTO)
    truncated = tuple(draw for draw in draws if draw.draw_number <= STAGE24_DISCOVERY_CUTOFF_DRAW)

    assert calculate_dataset_hash(discovery) == calculate_dataset_hash(truncated)


def test_holdout_requires_frozen_decision() -> None:
    with pytest.raises(ResearchValidationError, match="decision must be frozen"):
        evaluate_holdout_after_frozen_decision(
            _draws(),
            MINI_LOTO,
            frozen_decision={"strongest_signal": "multi_lag_recurrence_t2"},
        )
