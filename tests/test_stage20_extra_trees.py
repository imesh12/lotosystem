from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.extra_trees_evaluation import (
    CHALLENGER_KEEP,
    CHALLENGER_RETIRE,
    EXTRA_TREES_MODEL_NAME,
    benjamini_hochberg_adjust_p_values,
    make_extra_trees_model,
    register_stage20_experiment,
    run_stage20_extra_trees_evaluation,
)
from backend.app.research.history_import import write_canonical_history_csv
from backend.app.research.persistence import research_result_json


def _draws(
    lottery: LotteryDefinition,
    *,
    count: int = 34,
    start_number: int = 1000,
) -> tuple[HistoricalDraw, ...]:
    step = 3 if str(lottery.code) == "LOTO6" else 7
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
    draws: list[HistoricalDraw] = []
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
                date(2024, 1, 1) + timedelta(days=index * step),
                main,
                (bonus,),
            )
        )
    return tuple(draws)


def _config() -> ResearchConfig:
    return ResearchConfig(seed=123456, backtest_min_training_draws=10)


def test_extra_trees_model_uses_conservative_seeded_config() -> None:
    model = make_extra_trees_model(123456)

    assert model.n_estimators == 10
    assert model.max_depth == 6
    assert model.min_samples_leaf == 10
    assert model.random_state == 123456
    assert model.n_jobs == 1


def test_stage20_loto6_gap_only_evaluation_is_reproducible(tmp_path: Path) -> None:
    first = run_stage20_extra_trees_evaluation(
        _draws(LOTO6),
        LOTO6,
        _config(),
        tickets_per_draw=2,
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
    )
    second = run_stage20_extra_trees_evaluation(
        _draws(LOTO6),
        LOTO6,
        _config(),
        tickets_per_draw=2,
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
    )

    assert first == second
    assert first.feature_group == "gap_only"
    assert first.current_champion.model_name == "random_forest"
    assert first.extra_trees.model_name == EXTRA_TREES_MODEL_NAME
    assert first.extra_trees.comparison_vs_champion is not None
    assert first.leakage.lookahead_safe is True
    assert first.extra_trees.experiment_status in {CHALLENGER_KEEP, CHALLENGER_RETIRE}


def test_stage20_mini_loto_pair_only_evaluation(tmp_path: Path) -> None:
    result = run_stage20_extra_trees_evaluation(
        _draws(MINI_LOTO),
        MINI_LOTO,
        _config(),
        tickets_per_draw=2,
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
    )

    assert result.feature_group == "pair_only"
    assert result.current_champion.model_name == "logistic_regression"
    assert result.extra_trees.sample_size > 0
    assert result.extra_trees.tickets_per_draw == 2


def test_ledger_registration_expands_correction_scope(tmp_path: Path) -> None:
    result = run_stage20_extra_trees_evaluation(
        _draws(LOTO6),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "ledger.json",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["entries"].append(
        {
            "experiment_id": "OLD-EXP",
            "stage": "18",
            "lottery": "LOTO6",
            "dataset_hash": "old",
            "hypothesis": "old",
            "model": "old",
            "feature_group": "old",
            "comparison": "old",
            "raw_p_value": 0.01,
        }
    )
    ledger_path.write_text(research_result_json(ledger), encoding="utf-8")

    updated = register_stage20_experiment(
        result.extra_trees,
        lottery="LOTO6",
        dataset_hash=result.dataset_hash,
        ledger_path=ledger_path,
    )

    assert len(updated["entries"]) == 2
    assert all("ledger_adjusted_p_value" in entry for entry in updated["entries"])
    assert all("bh_exploratory_p_value" in entry for entry in updated["entries"])


def test_bh_adjustment_is_monotonic_and_bounded() -> None:
    adjusted = benjamini_hochberg_adjust_p_values({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_stage20_does_not_mutate_production_or_prospective_files(tmp_path: Path) -> None:
    prediction = tmp_path / "predictions" / "LOTO6" / "2132.json"
    prospective = tmp_path / "prospective" / "LOTO6" / "2131.json"
    prediction.parent.mkdir(parents=True)
    prospective.parent.mkdir(parents=True)
    prediction.write_text('{"immutable": true}', encoding="utf-8")
    prospective.write_text('{"immutable": true}', encoding="utf-8")
    before_prediction = prediction.read_bytes()
    before_prospective = prospective.read_bytes()

    run_stage20_extra_trees_evaluation(
        _draws(LOTO6),
        LOTO6,
        _config(),
        bootstrap_replications=10,
        ml_min_training_draws=12,
        ml_refit_interval=5,
        experiment_ledger_path=tmp_path / "exports" / "ledger.json",
    )

    assert prediction.read_bytes() == before_prediction
    assert prospective.read_bytes() == before_prospective


def test_stage20_cli_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "mini.csv"
    output_path = tmp_path / "stage20.json"
    write_canonical_history_csv(_draws(MINI_LOTO), data_path)

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
            "extra-trees-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["challenger_status"] in {CHALLENGER_KEEP, CHALLENGER_RETIRE}
    assert output_path.exists()
