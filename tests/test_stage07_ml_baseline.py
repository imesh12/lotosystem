from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.ml_baseline import (
    FEATURE_VERSION,
    build_training_dataset,
    build_walk_forward_feature_blocks,
    run_leakage_audit,
    run_stage07_ml_baseline,
    save_stage07_ml_baseline,
    tickets_from_scores,
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


def test_training_dataset_labels_and_feature_shape() -> None:
    blocks = build_walk_forward_feature_blocks(_loto6_draws(), LOTO6)
    x_rows, y_rows, training_dates = build_training_dataset(blocks, target_index=5)

    assert len(x_rows) == 4 * 43
    assert len(y_rows) == len(x_rows)
    assert sum(y_rows) == 4 * LOTO6.numbers_per_ticket
    assert all(len(row) == 12 for row in x_rows)
    assert training_dates[-1] < blocks[5].draw_date


def test_target_mutation_does_not_change_target_features() -> None:
    draws = list(_loto6_draws())
    original = build_walk_forward_feature_blocks(tuple(draws), LOTO6)[6]
    target = draws[6]
    draws[6] = HistoricalDraw(
        LOTO6,
        target.draw_number,
        target.draw_date,
        tuple(reversed(target.main_numbers)),
        target.bonus_numbers,
    )
    mutated = build_walk_forward_feature_blocks(tuple(draws), LOTO6)[6]

    assert tuple(row.features for row in original.rows) == tuple(
        row.features for row in mutated.rows
    )


def test_two_ticket_output_is_valid_and_distinct() -> None:
    scores = {number: float(100 - number) for number in range(1, 44)}
    tickets = tickets_from_scores(scores, LOTO6, tickets_per_draw=2)

    assert len(tickets) == 2
    assert tickets[0] != tickets[1]
    assert all(len(ticket) == LOTO6.numbers_per_ticket for ticket in tickets)


def test_stage07_models_are_reproducible(tmp_path: Path) -> None:
    first = run_stage07_ml_baseline(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
    )
    second = run_stage07_ml_baseline(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
    )
    output = tmp_path / "stage07.json"
    save_stage07_ml_baseline(first, output)

    assert first == second
    assert output.exists()
    assert first.feature_version == FEATURE_VERSION
    assert set(first.models) == {"logistic_regression", "random_forest"}


def test_leakage_audit_reports_safe_for_small_history() -> None:
    audit = run_leakage_audit(
        _loto6_draws(),
        LOTO6,
        seed=123456,
        ml_min_training_draws=8,
    )

    assert audit.lookahead_safe is True
    assert audit.training_dates_strictly_before_target is True
    assert audit.target_mutation_changes_features is False
    assert audit.future_mutation_changes_prediction is False


def test_stage07_supports_mini_loto() -> None:
    result = run_stage07_ml_baseline(
        _mini_draws(),
        MINI_LOTO,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
    )

    assert result.lottery == "MINI_LOTO"
    assert result.leakage.lookahead_safe is True
    assert all(model.metrics.tickets_evaluated > 0 for model in result.models.values())


def _write_draws_csv(path: Path, draws: tuple[HistoricalDraw, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "lottery",
                "draw_number",
                "draw_date",
                "main_numbers",
                "bonus_numbers",
            ),
        )
        writer.writeheader()
        for draw in draws:
            writer.writerow(
                {
                    "lottery": draw.lottery.code,
                    "draw_number": draw.draw_number,
                    "draw_date": draw.draw_date.isoformat(),
                    "main_numbers": " ".join(str(number) for number in draw.main_numbers),
                    "bonus_numbers": " ".join(str(number) for number in draw.bonus_numbers),
                }
            )


def test_stage07_cli_fixture_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "loto6.csv"
    _write_draws_csv(csv_path, _loto6_draws())
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
            "ml-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["feature_version"] == FEATURE_VERSION
    assert payload["leakage"]["lookahead_safe"] is True


def test_stage07_real_canonical_dataset_smoke_if_available() -> None:
    csv_path = Path("data/processed/mini_loto_history.csv")
    if not csv_path.exists():
        pytest.skip("canonical Mini Loto history file is not available")

    draws = load_draws_csv(csv_path, MINI_LOTO)
    result = run_stage07_ml_baseline(
        draws[:25],
        MINI_LOTO,
        _config(),
        bootstrap_replications=10,
        ml_min_training_draws=12,
    )

    assert result.dataset_hash
    assert result.models
