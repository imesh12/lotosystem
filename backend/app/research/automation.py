from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    HISTORY_UPDATE_NEW_RESULT,
    HISTORY_UPDATE_NO_NEW_RESULT,
    HISTORY_UPDATE_SOURCE_FAILURE,
    canonical_history_path,
)
from backend.app.research.notifications import (
    NOTIFICATION_ROOT,
    EmailConfig,
    EmailSender,
    email_config_from_env,
    notify_draw_processed,
    notify_source_failure,
)
from backend.app.research.operational_cycle import (
    OperationalCycleResult,
    cycle_result_payload,
    run_post_draw_cycle,
)
from backend.app.research.persistence import to_jsonable
from backend.app.research.production import (
    PREDICTION_ROOT,
    PREDICTION_STATUS_PENDING,
    generate_next_prediction,
    load_prediction_record,
    next_scheduled_draw_date,
    prediction_lottery_dir,
)
from backend.app.research.settings import SETTINGS_PATH, load_settings, lottery_settings
from backend.app.research.settlement import (
    FINANCIAL_STATUS_PAYOUT_PENDING,
    SETTLEMENT_ROOT,
    settle_evaluated_predictions,
)

AUTOMATION_ROOT = Path("data") / "automation"
AUTOMATION_TIMEZONE = "Asia/Tokyo"

ACTION_NO_ACTION = "NO_ACTION"
ACTION_CHECK_RESULT = "CHECK_RESULT"
ACTION_RESULT_PROCESSED = "RESULT_PROCESSED"
ACTION_SOURCE_FAILURE = "SOURCE_FAILURE"
ACTION_PREDICTION_CREATED = "PREDICTION_CREATED"


@dataclass(frozen=True, slots=True)
class AutomationConfig:
    timezone: str = AUTOMATION_TIMEZONE
    tickets_per_draw: int = 3
    result_check_hour: int = 21
    result_check_minute: int = 0
    retry_interval_minutes: int = 180
    stale_lock_minutes: int = 360


DEFAULT_AUTOMATION_CONFIG = AutomationConfig()
CycleRunner = Any


def automation_status(
    *,
    lottery: LotteryDefinition | None = None,
    prediction_root: str | Path = PREDICTION_ROOT,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    config: AutomationConfig = DEFAULT_AUTOMATION_CONFIG,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _aware_now(now, config)
    lotteries = _selected_lotteries(lottery)
    return {
        "current_time": current.isoformat(),
        "timezone": config.timezone,
        "lotteries": {
            str(selected.code): _lottery_status(
                selected,
                current,
                prediction_root=prediction_root,
                config=config,
            )
            for selected in lotteries
        },
        "financial_pending_count": _financial_pending_count(settlement_root),
    }


def run_automation_once(
    *,
    lottery: LotteryDefinition | None = None,
    config: AutomationConfig = DEFAULT_AUTOMATION_CONFIG,
    seed: int | None = None,
    prediction_root: str | Path = PREDICTION_ROOT,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    automation_root: str | Path = AUTOMATION_ROOT,
    notification_root: str | Path = NOTIFICATION_ROOT,
    email_config: EmailConfig | None = None,
    email_sender: EmailSender | None = None,
    settings_path: str | Path = SETTINGS_PATH,
    headed: bool = False,
    row_timeout_ms: int = 7_000,
    result_source_order: tuple[str, ...] | None = None,
    now: datetime | None = None,
    cycle_runner: CycleRunner | None = None,
) -> dict[str, Any]:
    current = _aware_now(now, config)
    root = Path(automation_root)
    lock_path = root / "automation.lock"
    _acquire_lock(lock_path, current, config)
    started = current
    errors: list[str] = []
    warnings: list[str] = []
    lottery_payloads: list[dict[str, Any]] = []
    operational_settings = load_settings(settings_path)
    try:
        for selected in _selected_lotteries(lottery):
            selected_settings = lottery_settings(selected, settings=operational_settings)
            planned = _lottery_status(
                selected,
                current,
                prediction_root=prediction_root,
                config=config,
            )
            if not selected_settings.enabled:
                payload = _disabled_payload(selected, planned)
            elif planned["next_action"] == ACTION_CHECK_RESULT:
                selected_config = _config_for_lottery(config, selected_settings.tickets_per_draw)
                payload = _run_due_cycle(
                    selected,
                    selected_config,
                    seed=seed,
                    prediction_root=prediction_root,
                    settlement_root=settlement_root,
                    headed=headed,
                    row_timeout_ms=row_timeout_ms,
                    result_source_order=result_source_order,
                    current=current,
                    cycle_runner=cycle_runner,
                )
            elif planned["next_action"] == ACTION_PREDICTION_CREATED:
                selected_config = _config_for_lottery(config, selected_settings.tickets_per_draw)
                payload = _create_future_prediction(
                    selected,
                    selected_config,
                    seed=seed,
                    prediction_root=prediction_root,
                    current=current,
                )
            else:
                payload = _no_action_payload(
                    selected,
                    planned,
                    prediction_root=prediction_root,
                    settlement_root=settlement_root,
                )
            payload = _with_notifications(
                selected,
                payload,
                planned,
                current=current,
                notification_root=notification_root,
                email_config=email_config or email_config_from_env(),
                email_sender=email_sender,
            )
            lottery_payloads.append(payload)
            errors.extend(payload.get("errors", ()))
            warnings.extend(payload.get("warnings", ()))
        completed = datetime.now(UTC).astimezone(_timezone(config.timezone))
        result = {
            "run_id": _run_id(started),
            "run_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "timezone": config.timezone,
            "lotteries": lottery_payloads,
            "warnings": tuple(warnings),
            "errors": tuple(errors),
        }
        result["record_path"] = str(_save_automation_record(result, root))
        return result
    finally:
        _release_lock(lock_path)


def _run_due_cycle(
    lottery: LotteryDefinition,
    config: AutomationConfig,
    *,
    seed: int | None,
    prediction_root: str | Path,
    settlement_root: str | Path,
    headed: bool,
    row_timeout_ms: int,
    result_source_order: tuple[str, ...] | None,
    current: datetime,
    cycle_runner: CycleRunner | None,
) -> dict[str, Any]:
    runner = cycle_runner or run_post_draw_cycle
    history_before = _history_summary(lottery)
    try:
        cycle: OperationalCycleResult = runner(
            lottery,
            ResearchConfig(seed=seed),
            tickets_per_draw=config.tickets_per_draw,
            prediction_root=prediction_root,
            settlement_root=settlement_root,
            headed=headed,
            row_timeout_ms=row_timeout_ms,
            result_source_order=result_source_order,
            started_at=current.astimezone(UTC),
        )
    except ResearchValidationError as exc:
        return {
            "lottery": str(lottery.code),
            "action": ACTION_SOURCE_FAILURE,
            "result_source_status": HISTORY_UPDATE_SOURCE_FAILURE,
            "history_update": {
                "previous_latest_draw": history_before["latest_draw_number"],
                "new_latest_draw": history_before["latest_draw_number"],
                "appended": 0,
            },
            "prediction_evaluation": {"evaluated": ()},
            "settlement": {"paths": ()},
            "next_prediction": None,
            "next_run_at": _retry_at(current, config).isoformat(),
            "warnings": (),
            "errors": (str(exc),),
        }

    cycle_payload = cycle_result_payload(cycle)
    status = cycle.history.update_status
    action = ACTION_RESULT_PROCESSED if status == HISTORY_UPDATE_NEW_RESULT else ACTION_CHECK_RESULT
    return {
        "lottery": str(lottery.code),
        "action": action,
        "result_source_status": status,
        "history_update": cycle_payload["history"],
        "prediction_evaluation": {"evaluated": cycle.evaluated_predictions},
        "settlement": {"paths": cycle.settlements},
        "next_prediction": cycle_payload["next_prediction"],
        "next_run_at": _next_run_after_cycle(cycle, current, config, status).isoformat(),
        "warnings": cycle.warnings,
        "errors": cycle.errors,
    }


def _with_notifications(
    lottery: LotteryDefinition,
    payload: dict[str, Any],
    planned: dict[str, Any],
    *,
    current: datetime,
    notification_root: str | Path,
    email_config: EmailConfig,
    email_sender: EmailSender | None,
) -> dict[str, Any]:
    notifications: list[dict[str, Any]] = []
    warnings = list(payload.get("warnings", ()))
    if payload["action"] == ACTION_RESULT_PROCESSED:
        next_prediction = payload.get("next_prediction") or {}
        next_prediction_path = next_prediction.get("record_path")
        for settlement_path in payload.get("settlement", {}).get("paths", ()):
            if not Path(settlement_path).exists():
                warnings.append(f"settlement not found for notification: {settlement_path}")
                continue
            try:
                record = notify_draw_processed(
                    settlement_path,
                    next_prediction_path=next_prediction_path,
                    notification_root=notification_root,
                    config=email_config,
                    sender=email_sender,
                    source_summary=payload.get("history_update", {}),
                )
                notifications.append(_notification_summary(record))
            except ResearchValidationError as exc:
                warnings.append(f"notification failed: {exc}")
    elif payload["action"] == ACTION_SOURCE_FAILURE:
        try:
            record = notify_source_failure(
                lottery,
                current_time=current,
                latest_history=planned["latest_history"],
                pending_prediction=planned["pending_prediction"],
                sources_attempted=tuple(
                    payload.get("history_update", {}).get("source_attempts", ())
                )
                or (
                    {
                        "source": "automated",
                        "result": ACTION_SOURCE_FAILURE,
                        "error": "; ".join(payload.get("errors", ())),
                    },
                ),
                notification_root=notification_root,
                config=email_config,
                sender=email_sender,
            )
            notifications.append(_notification_summary(record))
        except ResearchValidationError as exc:
            warnings.append(f"notification failed: {exc}")
    return {**payload, "notifications": tuple(notifications), "warnings": tuple(warnings)}


def _create_future_prediction(
    lottery: LotteryDefinition,
    config: AutomationConfig,
    *,
    seed: int | None,
    prediction_root: str | Path,
    current: datetime,
) -> dict[str, Any]:
    draws = _load_history(lottery)
    result = generate_next_prediction(
        draws,
        lottery,
        ResearchConfig(seed=seed),
        tickets_per_draw=config.tickets_per_draw,
        prediction_root=prediction_root,
        generated_at=current.astimezone(UTC),
    )
    action = ACTION_NO_ACTION if result.existing_record else ACTION_PREDICTION_CREATED
    next_check = _result_check_at(date.fromisoformat(result.record.target_draw_date), config)
    return {
        "lottery": str(lottery.code),
        "action": action,
        "result_source_status": None,
        "history_update": {
            "previous_latest_draw": result.record.latest_source_draw_number,
            "new_latest_draw": result.record.latest_source_draw_number,
            "appended": 0,
        },
        "prediction_evaluation": {"evaluated": ()},
        "settlement": {"paths": ()},
        "next_prediction": {
            "draw": result.record.target_draw_number,
            "target_date": result.record.target_draw_date,
            "status": result.record.status,
            "tickets": result.record.tickets_per_draw,
            "created": not result.existing_record,
            "record_path": result.record_path,
        },
        "next_run_at": next_check.isoformat(),
        "warnings": (),
        "errors": (),
    }


def _lottery_status(
    lottery: LotteryDefinition,
    current: datetime,
    *,
    prediction_root: str | Path,
    config: AutomationConfig,
) -> dict[str, Any]:
    history = _history_summary(lottery)
    pending = _pending_prediction_summary(prediction_root, lottery)
    if history["latest_draw_number"] is None:
        return {
            "latest_history": history,
            "pending_prediction": pending,
            "next_draw": None,
            "next_action": ACTION_NO_ACTION,
            "next_run_at": None,
            "warnings": ("canonical history is missing or empty",),
        }

    if pending is not None:
        target_date = date.fromisoformat(pending["target_draw_date"])
        check_at = _result_check_at(target_date, config)
        due = current >= check_at
        return {
            "latest_history": history,
            "pending_prediction": pending,
            "next_draw": {
                "draw_number": pending["target_draw_number"],
                "draw_date": pending["target_draw_date"],
            },
            "next_action": ACTION_CHECK_RESULT if due else ACTION_NO_ACTION,
            "next_run_at": (current if due else check_at).isoformat(),
            "warnings": (),
        }

    latest_date = date.fromisoformat(history["latest_draw_date"])
    target_date = next_scheduled_draw_date(latest_date, lottery)
    check_at = _result_check_at(target_date, config)
    action = ACTION_CHECK_RESULT if current >= check_at else ACTION_PREDICTION_CREATED
    return {
        "latest_history": history,
        "pending_prediction": None,
        "next_draw": {
            "draw_number": int(history["latest_draw_number"]) + 1,
            "draw_date": target_date.isoformat(),
        },
        "next_action": action,
        "next_run_at": (current if action == ACTION_CHECK_RESULT else current).isoformat(),
        "warnings": ()
        if action == ACTION_PREDICTION_CREATED
        else ("result check is due before creating any new future prediction",),
    }


def _no_action_payload(
    lottery: LotteryDefinition,
    planned: dict[str, Any],
    *,
    prediction_root: str | Path,
    settlement_root: str | Path,
) -> dict[str, Any]:
    reconciled_settlements = settle_evaluated_predictions(
        lottery,
        prediction_root=prediction_root,
        settlement_root=settlement_root,
    )
    return {
        "lottery": str(lottery.code),
        "action": ACTION_NO_ACTION,
        "result_source_status": None,
        "history_update": {
            "previous_latest_draw": planned["latest_history"]["latest_draw_number"],
            "new_latest_draw": planned["latest_history"]["latest_draw_number"],
            "appended": 0,
        },
        "prediction_evaluation": {"evaluated": ()},
        "settlement": {"paths": reconciled_settlements},
        "next_prediction": planned["pending_prediction"],
        "next_run_at": planned["next_run_at"],
        "warnings": planned["warnings"],
        "errors": (),
    }


def _disabled_payload(lottery: LotteryDefinition, planned: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "lottery": str(lottery.code),
        "action": ACTION_NO_ACTION,
        "result_source_status": None,
        "history_update": {
            "previous_latest_draw": planned["latest_history"]["latest_draw_number"],
            "new_latest_draw": planned["latest_history"]["latest_draw_number"],
            "appended": 0,
        },
        "prediction_evaluation": {"evaluated": ()},
        "settlement": {"paths": ()},
        "next_prediction": planned["pending_prediction"],
        "next_run_at": planned["next_run_at"],
        "warnings": planned["warnings"],
        "errors": (),
    }
    warnings = tuple((*payload["warnings"], "lottery automation is disabled in settings"))
    return {**payload, "settings_enabled": False, "warnings": warnings}


def _config_for_lottery(config: AutomationConfig, tickets_per_draw: int) -> AutomationConfig:
    return AutomationConfig(
        timezone=config.timezone,
        tickets_per_draw=tickets_per_draw,
        result_check_hour=config.result_check_hour,
        result_check_minute=config.result_check_minute,
        retry_interval_minutes=config.retry_interval_minutes,
        stale_lock_minutes=config.stale_lock_minutes,
    )


def _history_summary(lottery: LotteryDefinition) -> dict[str, Any]:
    path = canonical_history_path(lottery)
    if not path.exists():
        return {"path": str(path), "latest_draw_number": None, "latest_draw_date": None}
    draws = load_draws_csv(path, lottery)
    if not draws:
        return {"path": str(path), "latest_draw_number": None, "latest_draw_date": None}
    latest = draws[-1]
    return {
        "path": str(path),
        "latest_draw_number": latest.draw_number,
        "latest_draw_date": latest.draw_date.isoformat(),
    }


def _load_history(lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
    path = canonical_history_path(lottery)
    if not path.exists():
        raise ResearchValidationError(f"canonical history not found: {path}")
    return load_draws_csv(path, lottery)


def _pending_prediction_summary(
    prediction_root: str | Path,
    lottery: LotteryDefinition,
) -> dict[str, Any] | None:
    pending: list[dict[str, Any]] = []
    directory = prediction_lottery_dir(prediction_root, lottery)
    if not directory.exists():
        return None
    for path in sorted(directory.glob("*.json")):
        if path.name == "ledger.json":
            continue
        record = load_prediction_record(path)
        if record.status != PREDICTION_STATUS_PENDING:
            continue
        pending.append(
            {
                "draw": record.target_draw_number,
                "target_draw_number": record.target_draw_number,
                "target_date": record.target_draw_date,
                "target_draw_date": record.target_draw_date,
                "status": record.status,
                "tickets": record.tickets_per_draw,
                "record_path": str(path),
            }
        )
    return pending[0] if pending else None


def _selected_lotteries(lottery: LotteryDefinition | None) -> tuple[LotteryDefinition, ...]:
    return (LOTO6, MINI_LOTO) if lottery is None else (lottery,)


def _result_check_at(target_date: date, config: AutomationConfig) -> datetime:
    zone = _timezone(config.timezone)
    return datetime.combine(
        target_date,
        time(config.result_check_hour, config.result_check_minute),
        tzinfo=zone,
    )


def _next_run_after_cycle(
    cycle: OperationalCycleResult,
    current: datetime,
    config: AutomationConfig,
    update_status: str,
) -> datetime:
    if update_status in {HISTORY_UPDATE_NO_NEW_RESULT, HISTORY_UPDATE_SOURCE_FAILURE}:
        return _retry_at(current, config)
    if cycle.next_prediction is None:
        return current
    return _result_check_at(date.fromisoformat(cycle.next_prediction.target_date), config)


def _retry_at(current: datetime, config: AutomationConfig) -> datetime:
    return current + timedelta(minutes=config.retry_interval_minutes)


def _aware_now(now: datetime | None, config: AutomationConfig) -> datetime:
    zone = _timezone(config.timezone)
    value = now or datetime.now(zone)
    if value.tzinfo is None:
        return value.replace(tzinfo=zone)
    return value.astimezone(zone)


def _financial_pending_count(settlement_root: str | Path) -> int:
    root = Path(settlement_root)
    if not root.exists():
        return 0
    count = 0
    for path in root.glob("*/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("financial_status") == FINANCIAL_STATUS_PAYOUT_PENDING:
            count += 1
    return count


def _notification_summary(record: Any) -> dict[str, Any]:
    return {
        "notification_id": record.notification_id,
        "type": record.notification_type,
        "delivery_status": record.delivery_status,
        "attempt_count": record.attempt_count,
        "last_error": record.last_error,
    }


def _timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name == AUTOMATION_TIMEZONE:
            return timezone(timedelta(hours=9), name)
        raise


def _acquire_lock(path: Path, current: datetime, config: AutomationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        age = current.astimezone(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if age <= timedelta(minutes=config.stale_lock_minutes):
            raise ResearchValidationError(f"automation lock is active: {path}")
        path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ResearchValidationError(f"automation lock is active: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        lock_file.write(current.isoformat())


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _save_automation_record(payload: dict[str, Any], root: Path) -> Path:
    run_id = str(payload["run_id"])
    path = root / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_id(started: datetime) -> str:
    compact = started.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"AUTO-{compact}"
