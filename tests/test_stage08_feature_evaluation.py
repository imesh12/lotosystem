from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.feature_evaluation import (
    FEATURE_GROUPS,
    audit_feature_blocks,
    run_feature_leakage_audit,
    run_stage08_feature_evaluation,
)
from backend.app.research.ml_baseline import (
    FEATURE_NAMES_V1,
    FEATURE_NAMES_V2,
    FEATURE_VERSION_V2,
    _NumberFeatureState,
    build_walk_forward_feature_blocks,
)


def _loto6_draws(count: int = 14) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 20) + 1
        main = tuple(sorted(((start + offset * 5 - 1) % 43) + 1 for offset in range(6)))
        bonus = next(number for number in range(1, 44) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=LOTO6,
                draw_number=index + 1,
                draw_date=date(2026, 1, 1) + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _mini_draws(count: int = 14) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 16) + 1
        main = tuple(sorted(((start + offset * 4 - 1) % 31) + 1 for offset in range(5)))
        bonus = next(number for number in range(1, 32) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=MINI_LOTO,
                draw_number=index + 1,
                draw_date=date(2026, 1, 1) + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _config() -> ResearchConfig:
    return ResearchConfig(seed=123456, backtest_min_training_draws=3)


def test_v2_feature_formulas_are_deterministic_and_safe() -> None:
    state = _NumberFeatureState(LOTO6)
    state.add_draw(HistoricalDraw(LOTO6, 1, date(2026, 1, 1), (1, 2, 3, 4, 5, 6), (7,)))
    state.add_draw(HistoricalDraw(LOTO6, 2, date(2026, 1, 8), (1, 8, 9, 10, 11, 12), (13,)))

    first = state.feature_values_for_number(1)
    second = state.feature_values_for_number(1)

    assert first == second
    assert first["frequency_momentum_5_20"] == (2 / 5) - (2 / 20)
    assert first["gap_to_mean"] == 0.0
    assert first["previous_draw_presence"] == 1.0
    assert set(FEATURE_NAMES_V1).issubset(FEATURE_NAMES_V2)


def test_feature_group_composition_is_stable() -> None:
    assert FEATURE_GROUPS["v1_all"] == FEATURE_NAMES_V1
    assert FEATURE_GROUPS["v2_all"] == FEATURE_NAMES_V2
    assert "frequency_momentum_5_20" in FEATURE_GROUPS["v2_all"]
    assert "gap_to_mean" in FEATURE_GROUPS["v2_without_frequency_expansion"]


def test_feature_audit_detects_redundancy_and_temporal_rows() -> None:
    blocks = build_walk_forward_feature_blocks(_loto6_draws(), LOTO6, FEATURE_NAMES_V2)
    audit = audit_feature_blocks(blocks, FEATURE_NAMES_V2, FEATURE_VERSION_V2)

    assert audit.feature_count == len(FEATURE_NAMES_V2)
    assert audit.records["frequency_10"].missing_count == 0
    assert any(
        {pair.feature_a, pair.feature_b} == {"frequency_10", "recent_activity_10"}
        for pair in audit.correlated_pairs
    )
    assert audit.temporal_shifts


def test_feature_leakage_audit_for_v2() -> None:
    audit = run_feature_leakage_audit(
        _loto6_draws(),
        LOTO6,
        feature_names=FEATURE_NAMES_V2,
        seed=123456,
        ml_min_training_draws=8,
    )

    assert audit.lookahead_safe is True
    assert audit.target_mutation_changes_features is False
    assert audit.future_mutation_changes_prediction is False


def test_stage08_result_is_reproducible_and_has_ablations() -> None:
    first = run_stage08_feature_evaluation(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
    )
    second = run_stage08_feature_evaluation(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
    )

    assert first == second
    assert set(first.ablation_results) == set(FEATURE_GROUPS)
    assert first.leakage.lookahead_safe is True
    assert first.conclusion not in {"predictive_feature_proven", "winning_signal"}


def test_stage08_supports_mini_loto() -> None:
    result = run_stage08_feature_evaluation(
        _mini_draws(),
        MINI_LOTO,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
    )

    assert result.lottery == "MINI_LOTO"
    assert result.v2_audit.feature_count == len(FEATURE_NAMES_V2)
    assert result.leakage.lookahead_safe is True


def test_stage08_cli_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "loto6.csv"
    rows = [
        "lottery,draw_number,draw_date,main_numbers,bonus_numbers",
        *[
            ",".join(
                (
                    str(draw.lottery.code),
                    str(draw.draw_number),
                    draw.draw_date.isoformat(),
                    " ".join(str(number) for number in draw.main_numbers),
                    " ".join(str(number) for number in draw.bonus_numbers),
                )
            )
            for draw in _loto6_draws()
        ],
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--data",
            str(csv_path),
            "--seed",
            "123456",
            "--bootstrap-replications",
            "20",
            "--ml-min-training-draws",
            "8",
            "--ml-refit-interval",
            "3",
            "feature-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["leakage"]["lookahead_safe"] is True
    assert payload["v2_audit"]["feature_version"] == FEATURE_VERSION_V2
