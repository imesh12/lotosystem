from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import write_canonical_history_csv
from backend.app.research.pair_network import (
    PAIR_NETWORK_FEATURE_NAMES,
    PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW,
    PAIR_NETWORK_NEW_FEATURES,
    build_pair_network,
    build_pair_network_feature_blocks,
    pair_network_feature_values,
    run_pair_network_leakage_audit,
    run_stage22_pair_network_evaluation,
)


def _mini_draws(*, start_number: int = 1360, count: int = 41) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 20) + 1
        main = tuple(sorted(((start + offset * 4 - 1) % 31) + 1 for offset in range(5)))
        bonus = next(number for number in range(1, 32) if number not in main)
        draws.append(
            HistoricalDraw(
                MINI_LOTO,
                start_number + index,
                date(2025, 1, 7) + timedelta(days=index * 7),
                main,
                (bonus,),
            )
        )
    return tuple(draws)


def test_pair_network_uses_main_number_pairs_only_and_is_deterministic() -> None:
    draws = (
        HistoricalDraw(MINI_LOTO, 1, date(2025, 1, 7), (1, 2, 3, 4, 5), (6,)),
        HistoricalDraw(MINI_LOTO, 2, date(2025, 1, 14), (1, 2, 7, 8, 9), (3,)),
    )

    first = build_pair_network(draws, MINI_LOTO)
    second = build_pair_network(draws, MINI_LOTO)
    values = pair_network_feature_values(first, MINI_LOTO, 1)

    assert first == second
    assert first.pair_counts[(1, 2)] == 2
    assert all(6 not in pair for pair in first.pair_counts)
    assert values["weighted_degree"] == 8.0
    assert values["neighbor_strength_max"] == 2.0


def test_feature_blocks_exclude_target_and_future_draws() -> None:
    draws = (
        HistoricalDraw(MINI_LOTO, 1, date(2025, 1, 7), (1, 2, 3, 4, 5), (6,)),
        HistoricalDraw(MINI_LOTO, 2, date(2025, 1, 14), (1, 2, 7, 8, 9), (3,)),
        HistoricalDraw(MINI_LOTO, 3, date(2025, 1, 21), (1, 10, 11, 12, 13), (2,)),
    )

    blocks = build_pair_network_feature_blocks(draws, MINI_LOTO)
    target_two_number_one = next(row for row in blocks[1].rows if row.number == 1)
    target_three_number_one = next(row for row in blocks[2].rows if row.number == 1)

    assert target_two_number_one.features[0] == 4.0
    assert target_two_number_one.features[1] == 4.0
    assert target_three_number_one.features[1] == 8.0
    assert len(target_three_number_one.features) == len(PAIR_NETWORK_FEATURE_NAMES)


def test_pair_network_feature_group_augments_pair_only() -> None:
    assert PAIR_NETWORK_FEATURE_NAMES[0] == "pair_strength_rate"
    assert set(PAIR_NETWORK_NEW_FEATURES) < set(PAIR_NETWORK_FEATURE_NAMES)


def test_stage22_evaluation_excludes_1401_and_is_reproducible(tmp_path: Path) -> None:
    draws = (
        *_mini_draws(),
        HistoricalDraw(MINI_LOTO, 1401, date(2025, 10, 21), (1, 2, 3, 4, 5), (6,)),
    )

    first = run_stage22_pair_network_evaluation(
        draws,
        MINI_LOTO,
        ResearchConfig(seed=123456),
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
        preregistration_path=tmp_path / "prereg.json",
    )
    second = run_stage22_pair_network_evaluation(
        draws,
        MINI_LOTO,
        ResearchConfig(seed=123456),
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
        preregistration_path=tmp_path / "prereg.json",
    )

    assert first == second
    assert first.dataset_range["last_draw_number"] == PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW
    assert first.dataset_range["excluded_after_cutoff"] == 1
    assert first.champion["sample_size"] == first.challenger["sample_size"]
    assert first.leakage.lookahead_safe is True


def test_loto6_is_rejected_for_stage22(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        run_stage22_pair_network_evaluation(
            (),
            LOTO6,
            ResearchConfig(seed=123456),
            experiment_ledger_path=tmp_path / "ledger.json",
            preregistration_path=tmp_path / "prereg.json",
        )


def test_pair_network_leakage_audit_passes() -> None:
    audit = run_pair_network_leakage_audit(
        _mini_draws(),
        MINI_LOTO,
        seed=123456,
        ml_min_training_draws=12,
    )

    assert audit.lookahead_safe is True
    assert audit.training_dates_strictly_before_target is True
    assert audit.target_mutation_changes_features is False
    assert audit.future_mutation_changes_prediction is False


def test_stage22_ledger_includes_previous_hypotheses(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "v2-experiment-ledger-v1",
                "entries": [
                    {
                        "experiment_id": "OLD-MINI",
                        "lottery": "MINI_LOTO",
                        "raw_p_value": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_stage22_pair_network_evaluation(
        _mini_draws(),
        MINI_LOTO,
        ResearchConfig(seed=123456),
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=ledger,
        preregistration_path=tmp_path / "prereg.json",
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))

    assert result.governance["mini_loto_hypothesis_count"] == 2
    assert len(payload["entries"]) == 2
    assert all("ledger_adjusted_p_value" in entry for entry in payload["entries"])


def test_stage22_does_not_mutate_production_prospective_or_shadow(tmp_path: Path) -> None:
    watched = (
        tmp_path / "predictions" / "MINI_LOTO" / "1401.json",
        tmp_path / "prospective" / "summary.json",
        tmp_path / "shadow" / "registry.json",
    )
    for path in watched:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"immutable": true}', encoding="utf-8")
    before = {path: path.read_bytes() for path in watched}

    run_stage22_pair_network_evaluation(
        _mini_draws(),
        MINI_LOTO,
        ResearchConfig(seed=123456),
        bootstrap_replications=10,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "exports" / "ledger.json",
        preregistration_path=tmp_path / "exports" / "prereg.json",
    )

    assert {path: path.read_bytes() for path in watched} == before


def test_stage22_cli_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "mini.csv"
    output_path = tmp_path / "stage22.json"
    write_canonical_history_csv(_mini_draws(), data_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "MINI_LOTO",
            "--data",
            str(data_path),
            "--seed",
            "123456",
            "--bootstrap-replications",
            "10",
            "--ml-min-training-draws",
            "12",
            "--ml-refit-interval",
            "5",
            "--experiment-ledger",
            str(tmp_path / "ledger.json"),
            "--output",
            str(output_path),
            "pair-network-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["lookahead_safe"] is True
    assert output_path.exists()
