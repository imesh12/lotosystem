from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research import (
    CandidateStrategy,
    HistoricalDraw,
    ResearchConfig,
    build_candidate_features,
    calculate_dataset_hash,
    calculate_statistics,
    generate_candidates,
    generate_uniform_random_ticket,
    load_draws_csv,
    match_ticket,
    run_backtest,
    run_random_baseline_replications,
    run_research,
    score_candidate,
    validate_draw_sequence,
    validate_lottery_dataset,
)
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import (
    deterministic_research_payload,
    save_research_result,
    to_jsonable,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_draws() -> tuple[HistoricalDraw, ...]:
    return (
        HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 10, 20, 30), (40,)),
        HistoricalDraw(LOTO6, 2, date(2026, 1, 8), (2, 4, 6, 10, 22, 31), (41,)),
        HistoricalDraw(LOTO6, 3, date(2026, 1, 15), (3, 5, 7, 10, 23, 32), (42,)),
        HistoricalDraw(LOTO6, 4, date(2026, 1, 22), (1, 4, 7, 11, 24, 33), (43,)),
        HistoricalDraw(LOTO6, 5, date(2026, 1, 29), (8, 9, 10, 12, 25, 34), (35,)),
        HistoricalDraw(LOTO6, 6, date(2026, 2, 5), (1, 2, 10, 13, 26, 35), (36,)),
    )


@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig(
        frequency_windows=(3, 10),
        recent_window=3,
        candidate_pool_numbers=8,
        candidate_limit=5,
        backtest_min_training_draws=3,
        dataset_version="test-v001",
    )


def test_historical_draw_validation_rejects_invalid_record() -> None:
    with pytest.raises(ResearchValidationError, match="main numbers must be unique"):
        HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 1, 2, 3, 4, 5), (6,))


def test_validate_draw_sequence_rejects_unsorted_and_duplicates(
    sample_draws: tuple[HistoricalDraw, ...],
) -> None:
    with pytest.raises(ResearchValidationError, match="source rows are not chronological"):
        validate_draw_sequence((sample_draws[1], sample_draws[0]))

    with pytest.raises(ResearchValidationError, match="duplicate draw"):
        validate_draw_sequence((sample_draws[0], sample_draws[0]))


def test_mixed_lottery_dataset_is_rejected(sample_draws: tuple[HistoricalDraw, ...]) -> None:
    mixed = sample_draws + (
        HistoricalDraw(MINI_LOTO, 99, date(2026, 2, 12), (1, 2, 3, 4, 5), (6,)),
    )

    with pytest.raises(ResearchValidationError, match="mixed-lottery dataset"):
        validate_lottery_dataset(mixed, LOTO6)

    with pytest.raises(ResearchValidationError, match="mixed-lottery dataset"):
        calculate_statistics(mixed, LOTO6, ResearchConfig())


def test_csv_loader_supports_packed_numbers(tmp_path: Path) -> None:
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(
        "draw_number,draw_date,main_numbers,bonus_numbers,source\n"
        "1,2026-01-01,1 2 3 10 20 30,40,fixture\n",
        encoding="utf-8",
    )

    draws = load_draws_csv(csv_path, LOTO6)

    assert draws[0].main_numbers == (1, 2, 3, 10, 20, 30)
    assert draws[0].bonus_numbers == (40,)
    assert draws[0].source == "fixture"


def test_csv_loader_rejects_malformed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "lottery,draw_number,draw_date,n1,n2,n3,n4,n5,n6,bonus\n"
        "LOTO6,1,2026-01-01,1,2,3,4,5,47,6\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchValidationError, match="row 2:.*between 1 and 43"):
        load_draws_csv(csv_path, LOTO6)


def test_loto6_fixture_loads_and_hashes_deterministically() -> None:
    draws = load_draws_csv(FIXTURES_DIR / "loto6_history_sample.csv", LOTO6)

    assert len(draws) == 6
    assert calculate_dataset_hash(draws) == calculate_dataset_hash(tuple(reversed(draws)))


def test_mini_loto_fixture_loads() -> None:
    draws = load_draws_csv(FIXTURES_DIR / "mini_loto_history_sample.csv", MINI_LOTO)

    assert len(draws) == 6
    assert draws[0].main_numbers == (1, 2, 3, 10, 20)


def test_statistics_frequency_recency_distribution_pairs_gaps_and_transitions(
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    stats = calculate_statistics(sample_draws, LOTO6, config)

    assert stats.frequency[10].total_appearances == 5
    assert stats.frequency[10].window_counts == {3: 2, 10: 5}
    assert stats.recency[20].draws_since_last_seen == 5
    assert stats.recency[10].recent_count == 2
    assert 20 in stats.recency_summary.currently_absent_numbers
    assert stats.recency_summary.recently_frequent_numbers == (1, 10)
    assert 20 in stats.recency_summary.recently_inactive_numbers

    first_distribution = stats.distributions[0]
    assert first_distribution.total_sum == 66
    assert first_distribution.mean == 11
    assert first_distribution.minimum == 1
    assert first_distribution.maximum == 30
    assert first_distribution.median == 6.5
    assert first_distribution.odd_count == 2
    assert first_distribution.even_count == 4
    assert first_distribution.odd_even_pattern == "OEOEEE"
    assert first_distribution.low_count == 5
    assert first_distribution.high_count == 1
    assert first_distribution.low_high_pattern == "LLLLLH"
    assert first_distribution.consecutive_pair_count == 2
    assert first_distribution.consecutive_group_count == 1
    assert first_distribution.max_consecutive_group_length == 3

    assert stats.pairs[(1, 2)].occurrence_count == 2
    assert stats.gaps[1].gaps == (3, 2)
    assert stats.gaps[1].average_gap == 2.5
    assert stats.gaps[1].current_gap == 0
    assert stats.sum_statistics is not None
    assert stats.sum_statistics.minimum == 66
    assert stats.sum_statistics.maximum == 98
    assert stats.pattern_statistics.consecutive_pair_counts[2] == 2

    first_transition = stats.transitions.records[0]
    assert first_transition.overlap_count == 2
    assert first_transition.repeat_count == 2
    assert first_transition.new_number_count == 4
    assert first_transition.entering_numbers == (4, 6, 22, 31)
    assert stats.transitions.overlap_distribution[1] == 3
    assert stats.transitions.overlap_distribution[2] == 1


def test_feature_generation_and_scoring_are_explainable(
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    stats = calculate_statistics(sample_draws, LOTO6, config)
    features = build_candidate_features((1, 2, 3, 10, 20, 30), LOTO6, stats, config)
    score = score_candidate(features, LOTO6)

    assert features.frequency_total == 15
    assert features.recent_frequency_total == 5
    assert features.pair_strength >= 15
    assert score.frequency == 15
    assert score.recency == 5
    assert score.total == sum(
        (
            score.frequency,
            score.recency,
            score.gap,
            score.pair,
            score.distribution,
            score.pattern,
        )
    )


def test_candidate_generation_is_deterministic(
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    stats = calculate_statistics(sample_draws, LOTO6, config)

    first = generate_candidates(LOTO6, stats, config, CandidateStrategy.HYBRID)
    second = generate_candidates(LOTO6, stats, config, CandidateStrategy.HYBRID)
    baseline = generate_candidates(LOTO6, stats, config, CandidateStrategy.FIXED_BASELINE)

    assert first == second
    assert len(first) == 5
    assert baseline[0].numbers == (1, 2, 3, 4, 5, 6)
    assert all(candidate.strategy == CandidateStrategy.HYBRID for candidate in first)


def test_backtesting_uses_only_prior_draws(
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    result = run_backtest(sample_draws, LOTO6, config, CandidateStrategy.HYBRID)

    assert result.lookahead_safe is True
    assert result.strategy_metrics.total_evaluations == 3
    assert [step.target_draw_number for step in result.steps] == [4, 5, 6]
    assert [step.training_draw_count for step in result.steps] == [3, 4, 5]
    assert set(result.strategy_metrics.match_distribution) == set(range(7))
    assert result.baseline_strategy == CandidateStrategy.FIXED_BASELINE
    assert result.random_baseline.mean_matches >= 0


def test_backtesting_evaluates_multiple_candidates_and_date_range(
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    ranged_config = ResearchConfig(
        frequency_windows=config.frequency_windows,
        recent_window=config.recent_window,
        candidate_pool_numbers=8,
        candidate_limit=5,
        backtest_min_training_draws=3,
        backtest_candidate_count=2,
        evaluation_start=date(2026, 1, 29),
        evaluation_end=date(2026, 2, 5),
        baseline_replications=3,
        seed=123,
    )

    result = run_backtest(sample_draws, LOTO6, ranged_config, CandidateStrategy.HYBRID)

    assert [step.target_draw_number for step in result.steps] == [5, 6]
    assert all(len(step.strategy_evaluations) == 2 for step in result.steps)
    assert result.strategy_metrics.total_evaluations == 4
    assert result.lookahead_safe is True


def test_backtesting_handles_insufficient_history(sample_draws: tuple[HistoricalDraw, ...]) -> None:
    result = run_backtest(
        sample_draws[:2],
        LOTO6,
        ResearchConfig(backtest_min_training_draws=3, baseline_replications=2),
        CandidateStrategy.HYBRID,
    )

    assert result.strategy_metrics.total_evaluations == 0
    assert result.steps == ()


def test_research_pipeline_and_persistence_are_reproducible(
    tmp_path: Path,
    sample_draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> None:
    first = run_research(sample_draws, LOTO6, config, CandidateStrategy.HYBRID)
    second = run_research(sample_draws, LOTO6, config, CandidateStrategy.HYBRID)

    assert first == second
    assert first.dataset_hash == second.dataset_hash
    assert asdict(first)["configuration"] == asdict(second)["configuration"]

    output_path = tmp_path / "research.json"
    save_research_result(first, output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["result"]["dataset_hash"] == first.dataset_hash
    assert saved["result"]["strategy"] == CandidateStrategy.HYBRID.value
    assert deterministic_research_payload(first) == deterministic_research_payload(second)


def test_cli_validate_data_command(tmp_path: Path) -> None:
    csv_path = tmp_path / "draws.csv"
    csv_path.write_text(
        "draw_number,draw_date,n1,n2,n3,n4,n5,n6,bonus1\n1,2026-01-01,1,2,3,10,20,30,40\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--data",
            str(csv_path),
            "validate-data",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload == {"draw_count": 1, "status": "ok"}


def test_cli_backtest_command_for_mini_loto() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "MINI_LOTO",
            "--data",
            str(FIXTURES_DIR / "mini_loto_history_sample.csv"),
            "--seed",
            "123",
            "--baseline-replications",
            "3",
            "backtest",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["lookahead_safe"] is True
    assert payload["random_baseline"]["replications"][0]["seed"] == 123


def test_to_jsonable_handles_tuple_keys(
    sample_draws: tuple[HistoricalDraw, ...], config: ResearchConfig
) -> None:
    stats = calculate_statistics(sample_draws, LOTO6, config)
    payload = to_jsonable(stats.pairs)

    assert "[1, 2]" in payload


def test_prize_engine_loto6_match_states() -> None:
    draw = HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (3, 8, 15, 24, 31, 42), (19,))
    cases = (
        ((1, 2, 4, 5, 6, 7), 0, None),
        ((3, 1, 2, 4, 5, 6), 1, None),
        ((3, 8, 1, 2, 4, 5), 2, None),
        ((3, 8, 15, 1, 2, 4), 3, "5th"),
        ((3, 8, 15, 24, 1, 2), 4, "4th"),
        ((3, 8, 15, 24, 31, 1), 5, "3rd"),
        ((3, 8, 15, 24, 31, 19), 5, "2nd"),
        ((42, 31, 24, 15, 8, 3), 6, "1st"),
    )

    for ticket, expected_matches, expected_prize in cases:
        result = match_ticket(ticket, draw, LOTO6)
        assert result.main_match_count == expected_matches
        assert result.prize_name == expected_prize


def test_prize_engine_mini_loto_match_states() -> None:
    draw = HistoricalDraw(MINI_LOTO, 1, date(2026, 1, 1), (3, 8, 15, 24, 31), (19,))
    cases = (
        ((1, 2, 4, 5, 6), 0, None),
        ((3, 1, 2, 4, 5), 1, None),
        ((3, 8, 1, 2, 4), 2, None),
        ((3, 8, 15, 1, 2), 3, "4th"),
        ((3, 8, 15, 24, 1), 4, "3rd"),
        ((3, 8, 15, 24, 19), 4, "2nd"),
        ((31, 24, 15, 8, 3), 5, "1st"),
    )

    for ticket, expected_matches, expected_prize in cases:
        result = match_ticket(ticket, draw, MINI_LOTO)
        assert result.main_match_count == expected_matches
        assert result.prize_name == expected_prize


def test_seeded_uniform_random_baseline_is_reproducible() -> None:
    draws = load_draws_csv(FIXTURES_DIR / "loto6_history_sample.csv", LOTO6)
    config = ResearchConfig(seed=123, baseline_replications=5)

    first = run_random_baseline_replications(draws, LOTO6, config)
    second = run_random_baseline_replications(draws, LOTO6, config)
    different = run_random_baseline_replications(
        draws, LOTO6, ResearchConfig(seed=124, baseline_replications=5)
    )

    assert first == second
    assert first != different
    assert first.replications[0].evaluations[0].ticket == tuple(
        sorted(first.replications[0].evaluations[0].ticket)
    )


def test_random_ticket_validity() -> None:
    import random

    ticket = generate_uniform_random_ticket(MINI_LOTO, random.Random(1))

    assert len(ticket) == MINI_LOTO.numbers_per_ticket
    assert len(set(ticket)) == len(ticket)
    assert min(ticket) >= MINI_LOTO.number_min
    assert max(ticket) <= MINI_LOTO.number_max


def test_mini_loto_end_to_end_research() -> None:
    draws = load_draws_csv(FIXTURES_DIR / "mini_loto_history_sample.csv", MINI_LOTO)
    result = run_research(
        draws,
        MINI_LOTO,
        ResearchConfig(seed=456, baseline_replications=3, backtest_candidate_count=2),
        CandidateStrategy.HYBRID,
    )

    assert result.lottery == "MINI_LOTO"
    assert result.backtest.lookahead_safe is True
    assert result.backtest.random_baseline.replications[0].seed == 456
