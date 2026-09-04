from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    DEFAULT_HISTORY_START,
    HISTORY_UPDATE_SOURCE_FAILURE,
    HistoryUpdateResult,
    canonical_history_path,
    update_history_with_sources,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.production import (
    PREDICTION_ROOT,
    GeneratePredictionResult,
    evaluate_pending_predictions,
    generate_next_prediction,
    prediction_lottery_dir,
)
from backend.app.research.settlement import SETTLEMENT_ROOT, settle_evaluated_predictions


@dataclass(frozen=True, slots=True)
class CycleHistorySummary:
    previous_latest_draw: int | None
    new_latest_draw: int | None
    appended: int
    output_path: str
    update_status: str
    selected_source: str | None
    fallback_used: bool
    source_attempts: tuple[dict[str, str | None], ...]


@dataclass(frozen=True, slots=True)
class CycleNextPredictionSummary:
    draw: int
    target_date: str
    status: str
    tickets: int
    record_path: str
    created: bool


@dataclass(frozen=True, slots=True)
class OperationalCycleRecord:
    cycle_id: str
    lottery: str
    started_at: str
    completed_at: str | None
    history: CycleHistorySummary | None
    evaluated_predictions: tuple[int, ...]
    settlements: tuple[str, ...]
    next_prediction: CycleNextPredictionSummary | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    stage27: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class OperationalCycleResult:
    lottery: str
    cycle_id: str
    history: CycleHistorySummary
    evaluated_predictions: tuple[int, ...]
    settlements: tuple[str, ...]
    next_prediction: CycleNextPredictionSummary
    cycle_record_path: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    stage27: dict[str, Any] | None = None


HistoryUpdater = Callable[[LotteryDefinition], HistoryUpdateResult]


def run_post_draw_cycle(
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    history_path: str | Path | None = None,
    prediction_root: str | Path = PREDICTION_ROOT,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    tickets_per_draw: int = 3,
    headed: bool = False,
    row_timeout_ms: int = 7_000,
    history_start: date = DEFAULT_HISTORY_START,
    history_end: date | None = None,
    result_source_order: tuple[str, ...] | None = None,
    started_at: datetime | None = None,
    history_updater: HistoryUpdater | None = None,
    stage27_root: str | Path | None = None,
) -> OperationalCycleResult:
    started = (started_at or datetime.now(UTC)).astimezone(UTC)
    cycle_id = _cycle_id(lottery, started)
    destination = Path(history_path) if history_path else canonical_history_path(lottery)
    warnings: list[str] = []
    previous_latest: int | None = None
    try:
        before = _load_history_if_exists(destination, lottery)
        previous_latest = before[-1].draw_number if before else None
        updater = history_updater or (
            lambda selected_lottery: update_history_with_sources(
                selected_lottery,
                output_path=destination,
                start_date=history_start,
                end_date=history_end,
                headed=headed,
                row_timeout_ms=row_timeout_ms,
                source_order=result_source_order
                if result_source_order is not None
                else ("mizuho", "secondary"),
            )
        )
        update_result = updater(lottery)
        updated = load_draws_csv(destination, lottery)
        evaluation_result = evaluate_pending_predictions(
            updated,
            lottery,
            prediction_root=prediction_root,
        )
        settlement_paths = settle_evaluated_predictions(
            lottery,
            prediction_root=prediction_root,
            settlement_root=settlement_root,
        )
        prediction_result = generate_next_prediction(
            updated,
            lottery,
            config,
            tickets_per_draw=tickets_per_draw,
            prediction_root=prediction_root,
        )
        stage27 = (
            _stage27_cycle_payload(
                updated,
                lottery,
                config,
                prediction_root=prediction_root,
                stage27_root=stage27_root,
            )
            if str(lottery.code) == "MINI_LOTO"
            else None
        )
        if stage27 and stage27.get("warnings"):
            warnings.extend(str(warning) for warning in stage27["warnings"])
        history = CycleHistorySummary(
            previous_latest_draw=previous_latest,
            new_latest_draw=updated[-1].draw_number if updated else None,
            appended=update_result.appended_count,
            output_path=str(destination),
            update_status=update_result.update_status,
            selected_source=update_result.selected_source,
            fallback_used=update_result.fallback_used,
            source_attempts=_source_attempt_payload(update_result),
        )
        next_prediction = _next_prediction_summary(prediction_result)
        evaluated = _draw_numbers_from_paths(evaluation_result.evaluated_paths)
        record = OperationalCycleRecord(
            cycle_id=cycle_id,
            lottery=str(lottery.code),
            started_at=started.isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
            history=history,
            evaluated_predictions=evaluated,
            settlements=settlement_paths,
            next_prediction=next_prediction,
            stage27=stage27,
            errors=(),
            warnings=tuple(warnings),
        )
        record_path = save_cycle_record(record, prediction_root, lottery)
        return OperationalCycleResult(
            lottery=str(lottery.code),
            cycle_id=cycle_id,
            history=history,
            evaluated_predictions=evaluated,
            settlements=settlement_paths,
            next_prediction=next_prediction,
            stage27=stage27,
            cycle_record_path=str(record_path),
            errors=(),
            warnings=tuple(warnings),
        )
    except Exception as exc:
        history = (
            CycleHistorySummary(
                previous_latest_draw=previous_latest,
                new_latest_draw=previous_latest,
                appended=0,
                output_path=str(destination),
                update_status=HISTORY_UPDATE_SOURCE_FAILURE,
                selected_source=None,
                fallback_used=False,
                source_attempts=(),
            )
            if previous_latest is not None
            else None
        )
        record = OperationalCycleRecord(
            cycle_id=cycle_id,
            lottery=str(lottery.code),
            started_at=started.isoformat(),
            completed_at=datetime.now(UTC).isoformat(),
            history=history,
            evaluated_predictions=(),
            settlements=(),
            next_prediction=None,
            stage27=None,
            errors=(str(exc),),
            warnings=tuple(warnings),
        )
        save_cycle_record(record, prediction_root, lottery)
        if isinstance(exc, ResearchValidationError):
            raise
        raise ResearchValidationError(str(exc)) from exc


def save_cycle_record(
    record: OperationalCycleRecord,
    prediction_root: str | Path,
    lottery: LotteryDefinition,
) -> Path:
    directory = prediction_lottery_dir(prediction_root, lottery) / "cycles"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.cycle_id}.json"
    path.write_text(research_result_json(record), encoding="utf-8")
    return path


def _next_prediction_summary(result: GeneratePredictionResult) -> CycleNextPredictionSummary:
    return CycleNextPredictionSummary(
        draw=result.record.target_draw_number,
        target_date=result.record.target_draw_date,
        status=result.record.status,
        tickets=result.record.tickets_per_draw,
        record_path=result.record_path,
        created=not result.existing_record,
    )


def _load_history_if_exists(
    path: Path,
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    if not path.exists():
        return ()
    return load_draws_csv(path, lottery)


def _draw_numbers_from_paths(paths: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(Path(path).stem) for path in paths))


def _cycle_id(lottery: LotteryDefinition, started_at: datetime) -> str:
    compact = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"CYCLE-{lottery.code}-{compact}"


def cycle_result_payload(result: OperationalCycleResult) -> dict[str, Any]:
    return {
        "lottery": result.lottery,
        "cycle_id": result.cycle_id,
        "history": {
            "previous_latest_draw": result.history.previous_latest_draw,
            "new_latest_draw": result.history.new_latest_draw,
            "appended": result.history.appended,
            "update_status": result.history.update_status,
            "selected_source": result.history.selected_source,
            "fallback_used": result.history.fallback_used,
            "source_attempts": result.history.source_attempts,
        },
        "evaluated_predictions": result.evaluated_predictions,
        "settlements": result.settlements,
        "next_prediction": {
            "draw": result.next_prediction.draw,
            "target_date": result.next_prediction.target_date,
            "status": result.next_prediction.status,
            "tickets": result.next_prediction.tickets,
            "created": result.next_prediction.created,
        },
        "stage27": result.stage27,
        "cycle_record_path": result.cycle_record_path,
        "errors": result.errors,
        "warnings": result.warnings,
    }


def read_cycle_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_attempt_payload(result: HistoryUpdateResult) -> tuple[dict[str, str | None], ...]:
    return tuple(
        {
            "source": attempt.source,
            "result": attempt.result,
            "error": attempt.error,
        }
        for attempt in result.source_attempts
    )


def _stage27_cycle_payload(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    prediction_root: str | Path,
    stage27_root: str | Path | None,
) -> dict[str, Any] | None:
    if str(lottery.code) != "MINI_LOTO":
        return None
    try:
        from backend.app.research.stage27_prospective_signals import (
            run_stage27_cycle,
            stage27_payload,
        )

        seed = config.seed if config.seed is not None else 123456
        root = (
            Path(stage27_root)
            if stage27_root is not None
            else _default_stage27_root(prediction_root)
        )
        return stage27_payload(run_stage27_cycle(draws, lottery, seed=seed, root=root))
    except ResearchValidationError as exc:
        return {
            "experiment": "stage27_prospective_signal_tracking",
            "status": "ERROR",
            "error": str(exc),
            "warnings": (f"Stage 27 prospective tracking skipped: {exc}",),
        }


def _default_stage27_root(prediction_root: str | Path) -> Path:
    return Path(prediction_root).parent / "prospective" / "stage27"
