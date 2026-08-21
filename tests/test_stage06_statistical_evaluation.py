from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from backend.app.domain import LOTO6
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.statistical_evaluation import (
    ConfidenceInterval,
    bootstrap_confidence_interval,
    classify_conclusion,
    deterministic_experiment_id,
    holm_adjust_p_values,
    paired_permutation_p_value,
    run_stage06_statistical_evaluation,
    save_stage06_statistical_evaluation,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _draws() -> tuple[HistoricalDraw, ...]:
    return (
        HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 10, 20, 30), (40,)),
        HistoricalDraw(LOTO6, 2, date(2026, 1, 8), (2, 4, 6, 10, 22, 31), (41,)),
        HistoricalDraw(LOTO6, 3, date(2026, 1, 15), (3, 5, 7, 10, 23, 32), (42,)),
        HistoricalDraw(LOTO6, 4, date(2026, 1, 22), (1, 4, 7, 11, 24, 33), (43,)),
        HistoricalDraw(LOTO6, 5, date(2026, 1, 29), (8, 9, 10, 12, 25, 34), (35,)),
        HistoricalDraw(LOTO6, 6, date(2026, 2, 5), (1, 2, 10, 13, 26, 35), (36,)),
        HistoricalDraw(LOTO6, 7, date(2026, 2, 12), (1, 8, 14, 18, 28, 37), (39,)),
    )


def _config() -> ResearchConfig:
    return ResearchConfig(
        seed=123,
        baseline_replications=3,
        backtest_min_training_draws=3,
        candidate_pool_numbers=8,
        candidate_limit=4,
    )


def test_bootstrap_confidence_interval_is_reproducible_and_ordered() -> None:
    values = (0.0, 1.0, 1.0, 2.0)

    first = bootstrap_confidence_interval(
        values,
        seed=123,
        replications=100,
        confidence_level=0.95,
    )
    second = bootstrap_confidence_interval(
        values,
        seed=123,
        replications=100,
        confidence_level=0.95,
    )

    assert first == second
    assert first.lower <= first.upper
    assert first.confidence_level == 0.95


def test_paired_permutation_test_detects_directional_difference() -> None:
    p_value = paired_permutation_p_value((1.0, 1.0, 1.0, 1.0), seed=123, replications=200)

    assert 0 < p_value < 0.2


def test_holm_adjustment_is_monotonic() -> None:
    adjusted = holm_adjust_p_values({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_conclusion_classification_is_conservative() -> None:
    assert (
        classify_conclusion(
            adjusted_p_value=0.8,
            difference_ci=ConfidenceInterval(0.95, -0.1, 0.2),
            standardized_effect=0.01,
            stable_positive_periods=0,
            total_periods=4,
        )
        == "no_evidence"
    )
    assert (
        classify_conclusion(
            adjusted_p_value=0.01,
            difference_ci=ConfidenceInterval(0.95, 0.001, 0.02),
            standardized_effect=0.05,
            stable_positive_periods=4,
            total_periods=4,
        )
        == "statistically_detectable_small_effect"
    )


def test_deterministic_experiment_id_is_stable() -> None:
    first = deterministic_experiment_id(
        lottery="LOTO6",
        dataset_hash="abc",
        strategy="frequency",
        seed=123,
        bootstrap_replications=100,
        tickets_per_draw=2,
    )
    second = deterministic_experiment_id(
        lottery="LOTO6",
        dataset_hash="abc",
        strategy="frequency",
        seed=123,
        bootstrap_replications=100,
        tickets_per_draw=2,
    )

    assert first == second
    assert first.startswith("EXP-")


def test_stage06_result_structure_periods_and_payload(tmp_path: Path) -> None:
    result = run_stage06_statistical_evaluation(
        _draws(),
        LOTO6,
        _config(),
        tickets_per_draw=2,
        bootstrap_replications=100,
    )
    output_path = tmp_path / "stage06.json"
    save_stage06_statistical_evaluation(result, output_path)

    assert set(result.strategies) == {"balanced", "frequency", "hybrid", "pair", "recency"}
    assert result.multiple_comparison_method == "holm"
    assert output_path.exists()
    assert (tmp_path / "experiments").exists()
    for strategy_result in result.strategies.values():
        assert strategy_result.mean_matches.difference_ci.lower <= (
            strategy_result.mean_matches.difference_ci.upper
        )
        assert strategy_result.period_stability
        assert strategy_result.conclusion not in {"proven_predictive", "winning_model"}


def test_stage06_cli_fixture_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--data",
            str(FIXTURES_DIR / "loto6_history_sample.csv"),
            "--seed",
            "123",
            "--bootstrap-replications",
            "50",
            "statistical-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["multiple_comparison_method"] == "holm"


def test_stage06_real_canonical_dataset_smoke_if_available() -> None:
    csv_path = Path("data/processed/loto6_history.csv")
    if not csv_path.exists():
        pytest.skip("canonical LOTO6 history file is not available")

    draws = load_draws_csv(csv_path, LOTO6)
    result = run_stage06_statistical_evaluation(
        draws[:30],
        LOTO6,
        _config(),
        tickets_per_draw=2,
        bootstrap_replications=20,
    )

    assert result.dataset_hash
    assert result.strategies
