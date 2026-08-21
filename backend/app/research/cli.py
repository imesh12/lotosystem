from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from backend.app.domain.rules import get_lottery_definition
from backend.app.research.baseline_benchmark import (
    DEFAULT_STAGE05_REPLICATIONS,
    DEFAULT_STAGE05_SEED,
    DEFAULT_TICKETS_PER_DRAW,
    run_stage05_benchmark,
    save_stage05_benchmark_result,
)
from backend.app.research.candidates import generate_candidates
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.features import build_number_features
from backend.app.research.history_import import (
    DEFAULT_HISTORY_START,
    bootstrap_mizuho_history_with_browser,
    update_mizuho_history,
)
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    run_stage07_ml_baseline,
    save_stage07_ml_baseline,
)
from backend.app.research.persistence import save_research_result, to_jsonable
from backend.app.research.pipeline import run_research
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    run_stage06_statistical_evaluation,
    save_stage06_statistical_evaluation,
)
from backend.app.research.statistics import calculate_statistics


def main() -> None:
    parser = argparse.ArgumentParser(prog="loto-research")
    parser.add_argument("--lottery", default="LOTO6")
    parser.add_argument("--data")
    parser.add_argument("--window", type=int, action="append", dest="windows")
    parser.add_argument("--strategy", default=CandidateStrategy.HYBRID.value)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-replications", type=int)
    parser.add_argument(
        "--bootstrap-replications", type=int, default=DEFAULT_BOOTSTRAP_REPLICATIONS
    )
    parser.add_argument("--backtest-candidate-count", type=int, default=1)
    parser.add_argument("--tickets-per-draw", type=int, default=DEFAULT_TICKETS_PER_DRAW)
    parser.add_argument("--ml-min-training-draws", type=int, default=DEFAULT_ML_MIN_TRAINING_DRAWS)
    parser.add_argument("--ml-refit-interval", type=int, default=DEFAULT_ML_REFIT_INTERVAL)
    parser.add_argument("--evaluation-start")
    parser.add_argument("--evaluation-end")
    parser.add_argument("--history-start", default=DEFAULT_HISTORY_START.isoformat())
    parser.add_argument("--history-end")
    parser.add_argument("--output")
    parser.add_argument("--source-dir")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--browser-row-timeout-ms", type=int, default=7_000)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "update-history",
        "browser-bootstrap-history",
        "validate-data",
        "calculate-statistics",
        "generate-features",
        "generate-candidates",
        "backtest",
        "run-research",
        "baseline-benchmark",
        "statistical-evaluation",
        "ml-baseline",
    ):
        subparsers.add_parser(command)
    args = parser.parse_args()

    lottery = get_lottery_definition(args.lottery)

    if args.command == "update-history":
        try:
            result = update_mizuho_history(
                lottery,
                output_path=args.output,
                source_dir=args.source_dir,
                start_date=date.fromisoformat(args.history_start),
                end_date=date.fromisoformat(args.history_end) if args.history_end else None,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        payload = to_jsonable(asdict(result))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "browser-bootstrap-history":
        try:
            result = bootstrap_mizuho_history_with_browser(
                lottery,
                output_path=args.output,
                start_date=date.fromisoformat(args.history_start),
                end_date=date.fromisoformat(args.history_end) if args.history_end else None,
                headed=args.headed,
                row_timeout_ms=args.browser_row_timeout_ms,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        payload = to_jsonable(asdict(result))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if not args.data:
        parser.error(f"{args.command} requires --data")

    baseline_replications = (
        args.baseline_replications
        if args.baseline_replications is not None
        else (DEFAULT_STAGE05_REPLICATIONS if args.command == "baseline-benchmark" else 10)
    )
    seed = (
        args.seed
        if args.seed is not None
        else (DEFAULT_STAGE05_SEED if args.command == "baseline-benchmark" else None)
    )
    config = ResearchConfig(
        frequency_windows=tuple(args.windows or (10, 20, 50, 100)),
        seed=seed,
        baseline_replications=baseline_replications,
        backtest_candidate_count=args.backtest_candidate_count,
        evaluation_start=date.fromisoformat(args.evaluation_start)
        if args.evaluation_start
        else None,
        evaluation_end=date.fromisoformat(args.evaluation_end) if args.evaluation_end else None,
    )
    strategy = CandidateStrategy(args.strategy)
    try:
        draws = load_draws_csv(Path(args.data), lottery)
    except ResearchValidationError as exc:
        _exit_with_error(str(exc))

    if args.command == "validate-data":
        if not draws:
            _exit_with_error(f"no {lottery.code} draw records found in {args.data}")
        payload = {"status": "ok", "draw_count": len(draws)}
    elif args.command == "calculate-statistics":
        payload = to_jsonable(calculate_statistics(draws, lottery, config))
    elif args.command == "generate-features":
        payload = to_jsonable(build_number_features(calculate_statistics(draws, lottery, config)))
    elif args.command == "generate-candidates":
        stats = calculate_statistics(draws, lottery, config)
        payload = to_jsonable(generate_candidates(lottery, stats, config, strategy))
    elif args.command == "baseline-benchmark":
        try:
            result = run_stage05_benchmark(
                draws,
                lottery,
                config,
                tickets_per_draw=args.tickets_per_draw,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        if args.output:
            save_stage05_benchmark_result(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset.dataset_hash,
                "draws": result.dataset.draw_count,
                "tickets_evaluated": result.random_baseline.aggregate_metrics.tickets_evaluated,
            }
        else:
            payload = to_jsonable(result)
    elif args.command == "statistical-evaluation":
        try:
            result = run_stage06_statistical_evaluation(
                draws,
                lottery,
                config,
                tickets_per_draw=args.tickets_per_draw,
                bootstrap_replications=args.bootstrap_replications,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        if args.output:
            save_stage06_statistical_evaluation(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset_hash,
                "strategies": sorted(result.strategies),
            }
        else:
            payload = to_jsonable(result)
    elif args.command == "ml-baseline":
        try:
            result = run_stage07_ml_baseline(
                draws,
                lottery,
                config,
                tickets_per_draw=args.tickets_per_draw,
                bootstrap_replications=args.bootstrap_replications,
                ml_min_training_draws=args.ml_min_training_draws,
                ml_refit_interval=args.ml_refit_interval,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        if args.output:
            save_stage07_ml_baseline(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset_hash,
                "models": sorted(result.models),
                "lookahead_safe": result.leakage.lookahead_safe,
            }
        else:
            payload = to_jsonable(result)
    else:
        result = run_research(draws, lottery, config, strategy)
        payload = (
            to_jsonable(result.backtest) if args.command == "backtest" else to_jsonable(result)
        )
        if args.output:
            save_research_result(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset_hash,
                "evaluations": result.backtest.strategy_metrics.total_evaluations,
            }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _exit_with_error(message: str) -> None:
    print(json.dumps({"status": "error", "error": message}, indent=2, sort_keys=True))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
