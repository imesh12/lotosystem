from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.baseline_benchmark import (
    calculate_theoretical_sanity_check,
    generate_distinct_random_tickets,
    preflight_validate_benchmark_dataset,
    run_stage05_benchmark,
    run_two_ticket_random_baseline,
    save_stage05_benchmark_result,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import research_result_json


def _loto6_draws() -> tuple[HistoricalDraw, ...]:
    return (
        HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 10, 20, 30), (40,)),
        HistoricalDraw(LOTO6, 2, date(2026, 1, 8), (2, 4, 6, 10, 22, 31), (41,)),
        HistoricalDraw(LOTO6, 3, date(2026, 1, 15), (3, 5, 7, 10, 23, 32), (42,)),
        HistoricalDraw(LOTO6, 4, date(2026, 1, 22), (1, 4, 7, 11, 24, 33), (43,)),
        HistoricalDraw(LOTO6, 5, date(2026, 1, 29), (8, 9, 10, 12, 25, 34), (35,)),
        HistoricalDraw(LOTO6, 6, date(2026, 2, 5), (1, 2, 10, 13, 26, 35), (36,)),
    )


def test_stage05_random_tickets_are_exactly_two_distinct_and_valid() -> None:
    tickets = generate_distinct_random_tickets(LOTO6, random.Random(123), 2)

    assert len(tickets) == 2
    assert len(set(tickets)) == 2
    assert all(len(ticket) == LOTO6.numbers_per_ticket for ticket in tickets)
    assert all(len(set(ticket)) == len(ticket) for ticket in tickets)
    assert all(
        min(ticket) >= LOTO6.number_min and max(ticket) <= LOTO6.number_max for ticket in tickets
    )


def test_stage05_mini_loto_random_tickets_are_valid() -> None:
    tickets = generate_distinct_random_tickets(MINI_LOTO, random.Random(456), 2)

    assert len(tickets) == 2
    assert all(len(ticket) == MINI_LOTO.numbers_per_ticket for ticket in tickets)
    assert all(max(ticket) <= MINI_LOTO.number_max for ticket in tickets)


def test_stage05_random_baseline_reproducibility_and_ticket_count() -> None:
    draws = _loto6_draws()

    first = run_two_ticket_random_baseline(
        draws, LOTO6, seed=123, replications=4, tickets_per_draw=2
    )
    second = run_two_ticket_random_baseline(
        draws, LOTO6, seed=123, replications=4, tickets_per_draw=2
    )
    different = run_two_ticket_random_baseline(
        draws,
        LOTO6,
        seed=124,
        replications=4,
        tickets_per_draw=2,
    )

    assert first == second
    assert first != different
    assert first.aggregate_metrics.tickets_evaluated == len(draws) * 2 * 4


def test_stage05_prize_category_aggregation() -> None:
    draw = HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 4, 5, 6), (7,))
    result = run_two_ticket_random_baseline(
        (draw,), LOTO6, seed=1, replications=1, tickets_per_draw=2
    )

    assert set(result.aggregate_metrics.prize_category_counts) == {
        tier.name for tier in LOTO6.prize_tiers
    }


def test_stage05_theoretical_probability_calculation() -> None:
    sanity = calculate_theoretical_sanity_check(LOTO6, 0.0, 0.0)

    assert sum(sanity.match_probabilities.values()) == pytest.approx(1.0)
    assert sanity.expected_average_matches == pytest.approx(36 / 43)
    assert sanity.prize_qualified_rate > 0


def test_stage05_benchmark_structure_and_deterministic_serialization(tmp_path: Path) -> None:
    config = ResearchConfig(
        seed=123,
        baseline_replications=3,
        backtest_min_training_draws=3,
        candidate_pool_numbers=8,
        candidate_limit=4,
    )

    first = run_stage05_benchmark(_loto6_draws(), LOTO6, config, tickets_per_draw=2)
    second = run_stage05_benchmark(_loto6_draws(), LOTO6, config, tickets_per_draw=2)
    output_path = tmp_path / "stage05.json"
    save_stage05_benchmark_result(first, output_path)

    assert first == second
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"].startswith(
        "stage05"
    )
    assert research_result_json(first) == research_result_json(second)
    assert first.random_baseline.aggregate_metrics.tickets_evaluated == 36


def test_stage05_strategy_evaluation_uses_two_ticket_walk_forward_semantics() -> None:
    config = ResearchConfig(
        seed=123,
        baseline_replications=2,
        backtest_min_training_draws=3,
        candidate_pool_numbers=8,
        candidate_limit=4,
    )

    result = run_stage05_benchmark(_loto6_draws(), LOTO6, config, tickets_per_draw=2)

    for benchmark in result.strategy_metrics.values():
        assert benchmark.lookahead_safe is True
        assert benchmark.metrics.draws_evaluated == 3
        assert benchmark.metrics.tickets_evaluated == 6


def test_stage05_preflight_rejects_missing_draw_number_gap() -> None:
    draws = (
        HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 3, date(2026, 1, 8), (8, 9, 10, 11, 12, 13), (14,)),
    )

    with pytest.raises(ResearchValidationError, match="missing draw numbers"):
        preflight_validate_benchmark_dataset(draws, LOTO6)


def test_stage05_real_canonical_dataset_preflight_if_available() -> None:
    csv_path = Path("data/processed/loto6_history.csv")
    if not csv_path.exists():
        pytest.skip("canonical LOTO6 history file is not available")

    draws = load_draws_csv(csv_path, LOTO6)
    preflight = preflight_validate_benchmark_dataset(draws, LOTO6)

    assert preflight.draw_count > 0
    assert preflight.missing_draw_numbers == ()
