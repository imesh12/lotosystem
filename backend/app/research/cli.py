from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from backend.app.domain.rules import get_lottery_definition
from backend.app.research.automation import (
    AutomationConfig,
    automation_status,
    run_automation_once,
)
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
from backend.app.research.feature_evaluation import (
    run_stage08_feature_evaluation,
    save_stage08_feature_evaluation,
)
from backend.app.research.features import build_number_features
from backend.app.research.history_import import (
    DEFAULT_HISTORY_START,
    append_manual_result,
    bootstrap_mizuho_history_with_browser,
    update_mizuho_history,
)
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    run_stage07_ml_baseline,
    save_stage07_ml_baseline,
)
from backend.app.research.notifications import (
    notification_status,
    send_pending_notifications,
    send_test_email,
)
from backend.app.research.operational_cycle import (
    cycle_result_payload,
    run_post_draw_cycle,
)
from backend.app.research.persistence import save_research_result, to_jsonable
from backend.app.research.pipeline import run_research
from backend.app.research.portfolio_evaluation import (
    run_stage09_portfolio_evaluation,
    save_stage09_portfolio_evaluation,
)
from backend.app.research.production import (
    evaluate_pending_predictions,
    generate_next_prediction,
)
from backend.app.research.settlement import (
    add_manual_payout,
    financial_summary,
)
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
    parser.add_argument("--prediction-root")
    parser.add_argument("--settlement-root")
    parser.add_argument("--automation-root")
    parser.add_argument("--notification-root")
    parser.add_argument("--result-source", action="append", dest="result_sources")
    parser.add_argument("--result-check-hour", type=int)
    parser.add_argument("--retry-interval-minutes", type=int)
    parser.add_argument("--draw-number", type=int)
    parser.add_argument("--draw-date")
    parser.add_argument("--numbers")
    parser.add_argument("--bonus")
    parser.add_argument("--confirm-manual", action="store_true")
    parser.add_argument("--tier")
    parser.add_argument("--payout", type=int)
    parser.add_argument("--winners-count", type=int)
    parser.add_argument("--date")
    parser.add_argument("--month")
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
        "feature-evaluation",
        "portfolio-evaluation",
        "generate-next",
        "evaluate-predictions",
        "add-result",
        "add-payout",
        "financial-summary",
        "run-cycle",
        "automation-status",
        "auto-run",
        "notification-status",
        "send-pending-notifications",
        "test-email",
    ):
        subparsers.add_parser(command)
    args = parser.parse_args()

    all_lottery_commands = {
        "financial-summary",
        "automation-status",
        "auto-run",
        "notification-status",
        "send-pending-notifications",
        "test-email",
    }
    lottery = (
        None
        if args.command in all_lottery_commands and args.lottery == "ALL"
        else get_lottery_definition(args.lottery)
    )

    if args.command == "update-history":
        assert lottery is not None
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
        assert lottery is not None
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

    if args.command == "run-cycle":
        assert lottery is not None
        baseline_replications = (
            args.baseline_replications if args.baseline_replications is not None else 10
        )
        config = ResearchConfig(
            frequency_windows=tuple(args.windows or (10, 20, 50, 100)),
            seed=args.seed,
            baseline_replications=baseline_replications,
        )
        try:
            result = run_post_draw_cycle(
                lottery,
                config,
                tickets_per_draw=args.tickets_per_draw,
                prediction_root=args.prediction_root or "data/predictions",
                settlement_root=args.settlement_root or "data/settlements",
                headed=args.headed,
                row_timeout_ms=args.browser_row_timeout_ms,
                history_start=date.fromisoformat(args.history_start),
                history_end=date.fromisoformat(args.history_end) if args.history_end else None,
                result_source_order=tuple(args.result_sources) if args.result_sources else None,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(cycle_result_payload(result), indent=2, sort_keys=True, default=str))
        return

    if args.command == "add-result":
        assert lottery is not None
        try:
            result = append_manual_result(
                lottery,
                draw_number=_required_int(args.draw_number, "--draw-number"),
                draw_date=date.fromisoformat(_required_text(args.draw_date, "--draw-date")),
                main_numbers=_parse_cli_numbers(
                    _required_text(args.numbers, "--numbers"),
                    expected_count=lottery.numbers_per_ticket,
                ),
                bonus_numbers=_parse_cli_numbers(
                    _required_text(args.bonus, "--bonus"),
                    expected_count=lottery.bonus_numbers,
                ),
                output_path=args.output,
                confirmed=args.confirm_manual,
            )
        except (ValueError, ResearchValidationError) as exc:
            _exit_with_error(str(exc))
        payload = to_jsonable(asdict(result))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "add-payout":
        assert lottery is not None
        try:
            result = add_manual_payout(
                lottery,
                draw_number=_required_int(args.draw_number, "--draw-number"),
                prize_tier=_required_text(args.tier, "--tier"),
                payout_yen=_required_int(args.payout, "--payout"),
                winners_count=args.winners_count,
                settlement_root=args.settlement_root or "data/settlements",
                confirmed=args.confirm_manual,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        payload = to_jsonable(result)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "financial-summary":
        try:
            payload = financial_summary(
                settlement_root=args.settlement_root or "data/settlements",
                lottery=lottery,
                on_date=date.fromisoformat(args.date) if args.date else None,
                month=args.month,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "automation-status":
        try:
            payload = automation_status(
                lottery=lottery,
                prediction_root=args.prediction_root or "data/predictions",
                settlement_root=args.settlement_root or "data/settlements",
                config=_automation_config(args),
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "auto-run":
        try:
            payload = run_automation_once(
                lottery=lottery,
                config=_automation_config(args),
                seed=args.seed,
                prediction_root=args.prediction_root or "data/predictions",
                settlement_root=args.settlement_root or "data/settlements",
                automation_root=args.automation_root or "data/automation",
                notification_root=args.notification_root or "data/notifications",
                headed=args.headed,
                row_timeout_ms=args.browser_row_timeout_ms,
                result_source_order=tuple(args.result_sources) if args.result_sources else None,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "notification-status":
        try:
            payload = notification_status(
                notification_root=args.notification_root or "data/notifications",
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "send-pending-notifications":
        try:
            payload = send_pending_notifications(
                notification_root=args.notification_root or "data/notifications",
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    if args.command == "test-email":
        try:
            payload = send_test_email(
                notification_root=args.notification_root or "data/notifications",
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True, default=str))
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
    elif args.command == "feature-evaluation":
        try:
            result = run_stage08_feature_evaluation(
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
            save_stage08_feature_evaluation(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset_hash,
                "feature_groups": sorted(result.ablation_results),
                "lookahead_safe": result.leakage.lookahead_safe,
                "conclusion": result.conclusion,
            }
        else:
            payload = to_jsonable(result)
    elif args.command == "portfolio-evaluation":
        try:
            result = run_stage09_portfolio_evaluation(
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
            save_stage09_portfolio_evaluation(result, args.output)
            payload = {
                "status": "ok",
                "output": args.output,
                "lottery": str(lottery.code),
                "dataset_hash": result.dataset_hash,
                "methods": sorted(result.method_results),
                "lookahead_safe": result.leakage.lookahead_safe,
                "conclusion": result.conclusion,
            }
        else:
            payload = to_jsonable(result)
    elif args.command == "generate-next":
        try:
            result = generate_next_prediction(
                draws,
                lottery,
                config,
                tickets_per_draw=args.tickets_per_draw,
                prediction_root=args.prediction_root or "data/predictions",
                ml_min_training_draws=args.ml_min_training_draws,
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
        payload = {
            "status": result.record.status,
            "lottery": str(lottery.code),
            "target_draw_number": result.record.target_draw_number,
            "target_draw_date": result.record.target_draw_date,
            "tickets": [ticket.numbers for ticket in result.record.tickets],
            "record_path": result.record_path,
            "existing_record": result.existing_record,
            "lookahead_safe": result.lookahead_safe,
        }
    elif args.command == "evaluate-predictions":
        try:
            result = evaluate_pending_predictions(
                draws,
                lottery,
                prediction_root=args.prediction_root or "data/predictions",
            )
        except ResearchValidationError as exc:
            _exit_with_error(str(exc))
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


def _required_text(value: str | None, name: str) -> str:
    if not value:
        raise ResearchValidationError(f"{name} is required")
    return value


def _required_int(value: int | None, name: str) -> int:
    if value is None:
        raise ResearchValidationError(f"{name} is required")
    return value


def _parse_cli_numbers(value: str, *, expected_count: int) -> tuple[int, ...]:
    numbers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(numbers) != expected_count:
        raise ResearchValidationError(f"expected {expected_count} numbers, found {len(numbers)}")
    return numbers


def _automation_config(args: argparse.Namespace) -> AutomationConfig:
    config = AutomationConfig(tickets_per_draw=args.tickets_per_draw)
    if args.result_check_hour is None and args.retry_interval_minutes is None:
        return config
    return AutomationConfig(
        tickets_per_draw=args.tickets_per_draw,
        result_check_hour=args.result_check_hour
        if args.result_check_hour is not None
        else config.result_check_hour,
        retry_interval_minutes=args.retry_interval_minutes
        if args.retry_interval_minutes is not None
        else config.retry_interval_minutes,
    )


if __name__ == "__main__":
    main()
