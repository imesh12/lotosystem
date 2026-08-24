from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO, get_lottery_definition
from backend.app.research.automation import automation_status
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import canonical_history_path
from backend.app.research.notifications import notification_status
from backend.app.research.production import (
    PREDICTION_ROOT,
    PREDICTION_STATUS_PENDING,
    PredictionRecord,
    load_prediction_record,
    prediction_lottery_dir,
    prediction_record_path,
)
from backend.app.research.settings import (
    OperationalSettings,
    load_settings,
    settings_payload,
)
from backend.app.research.settlement import (
    SETTLEMENT_ROOT,
    DrawSettlement,
    financial_summary,
    load_settlement,
    settlement_lottery_dir,
    settlement_path,
)

DEFAULT_HISTORY_LIMIT = 20
MAX_HISTORY_LIMIT = 100
OPERATION_PREDICTION_ROOT = PREDICTION_ROOT
OPERATION_SETTLEMENT_ROOT = SETTLEMENT_ROOT


def status_payload(
    *,
    settings: OperationalSettings | None = None,
    prediction_root: str | Path | None = None,
    settlement_root: str | Path | None = None,
) -> dict[str, Any]:
    active_settings = settings or load_settings()
    prediction_base = prediction_root or OPERATION_PREDICTION_ROOT
    settlement_base = settlement_root or OPERATION_SETTLEMENT_ROOT
    today = date.today()
    month = today.strftime("%Y-%m")
    automation = _public_payload(
        automation_status(prediction_root=prediction_base, settlement_root=settlement_base)
    )
    return {
        "system": {
            "current_jst_time": automation["current_time"],
            "automation": automation,
            "email": {"enabled": active_settings.email_enabled},
        },
        "LOTO6": _lottery_dashboard_payload(
            LOTO6,
            prediction_root=prediction_base,
            settlement_root=settlement_base,
            automation=automation["lotteries"].get("LOTO6"),
        ),
        "MINI_LOTO": _lottery_dashboard_payload(
            MINI_LOTO,
            prediction_root=prediction_base,
            settlement_root=settlement_base,
            automation=automation["lotteries"].get("MINI_LOTO"),
        ),
        "financial": {
            "today": financial_summary(settlement_root=settlement_base, on_date=today),
            "current_month": financial_summary(settlement_root=settlement_base, month=month),
            "all_time": financial_summary(settlement_root=settlement_base),
        },
        "warnings": _status_warnings(automation),
    }


def lotteries_payload(settings: OperationalSettings | None = None) -> dict[str, Any]:
    active = settings or load_settings()
    return {
        "lotteries": [
            {
                "code": str(LOTO6.code),
                "name": LOTO6.name,
                "draw_schedule": LOTO6.draw_schedule,
                "enabled": active.loto6.enabled,
                "tickets_per_draw": active.loto6.tickets_per_draw,
            },
            {
                "code": str(MINI_LOTO.code),
                "name": MINI_LOTO.name,
                "draw_schedule": MINI_LOTO.draw_schedule,
                "enabled": active.mini_loto.enabled,
                "tickets_per_draw": active.mini_loto.tickets_per_draw,
            },
        ]
    }


def latest_lottery_payload(
    lottery: LotteryDefinition,
    *,
    prediction_root: str | Path | None = None,
    settlement_root: str | Path | None = None,
) -> dict[str, Any]:
    draws = _load_history(lottery)
    latest = draws[-1] if draws else None
    if latest is None:
        return {"lottery": str(lottery.code), "latest": None}
    prediction_base = prediction_root or OPERATION_PREDICTION_ROOT
    settlement_base = settlement_root or OPERATION_SETTLEMENT_ROOT
    settlement = _load_settlement_if_exists(settlement_base, lottery, latest.draw_number)
    prediction = _load_prediction_if_exists(prediction_base, lottery, latest.draw_number)
    return {
        "lottery": str(lottery.code),
        "draw_number": latest.draw_number,
        "draw_date": latest.draw_date.isoformat(),
        "main_numbers": latest.main_numbers,
        "bonus_numbers": latest.bonus_numbers,
        "prediction": _prediction_summary(prediction) if prediction else None,
        "prediction_available": prediction is not None,
        "ticket_results": () if settlement is None else _settlement_ticket_results(settlement),
        "paper_financial": None if settlement is None else _settlement_financial(settlement),
        "settlement_status": None if settlement is None else settlement.financial_status,
    }


def next_prediction_payload(
    lottery: LotteryDefinition,
    *,
    prediction_root: str | Path | None = None,
) -> dict[str, Any]:
    record = _latest_pending_prediction(prediction_root or OPERATION_PREDICTION_ROOT, lottery)
    return {"lottery": str(lottery.code), "pending_prediction": _prediction_summary(record)}


def history_payload(
    lottery: LotteryDefinition,
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
    offset: int = 0,
    prediction_root: str | Path | None = None,
    settlement_root: str | Path | None = None,
) -> dict[str, Any]:
    if limit <= 0 or limit > MAX_HISTORY_LIMIT:
        raise ResearchValidationError(f"limit must be between 1 and {MAX_HISTORY_LIMIT}")
    if offset < 0:
        raise ResearchValidationError("offset must be non-negative")
    prediction_base = prediction_root or OPERATION_PREDICTION_ROOT
    settlement_base = settlement_root or OPERATION_SETTLEMENT_ROOT
    draws = tuple(reversed(_load_history(lottery)))
    selected = draws[offset : offset + limit]
    rows = tuple(
        _history_row(
            draw,
            prediction_root=prediction_base,
            settlement_root=settlement_base,
        )
        for draw in selected
    )
    return {
        "lottery": str(lottery.code),
        "limit": limit,
        "offset": offset,
        "total": len(draws),
        "rows": rows,
    }


def financial_summary_payload(
    *,
    lottery_code: str = "ALL",
    period: str = "all_time",
    settlement_root: str | Path | None = None,
) -> dict[str, Any]:
    settlement_base = settlement_root or OPERATION_SETTLEMENT_ROOT
    lottery = None if lottery_code == "ALL" else get_lottery_definition(lottery_code)
    today = date.today()
    if period == "today":
        return financial_summary(settlement_root=settlement_base, lottery=lottery, on_date=today)
    if period == "month":
        return financial_summary(
            settlement_root=settlement_base,
            lottery=lottery,
            month=today.strftime("%Y-%m"),
        )
    if period == "all_time":
        return financial_summary(settlement_root=settlement_base, lottery=lottery)
    raise ResearchValidationError("period must be today, month, or all_time")


def notification_status_payload() -> dict[str, Any]:
    return _public_payload(notification_status())


def settings_api_payload(settings: OperationalSettings | None = None) -> dict[str, Any]:
    return settings_payload(settings or load_settings())


def _lottery_dashboard_payload(
    lottery: LotteryDefinition,
    *,
    prediction_root: str | Path,
    settlement_root: str | Path,
    automation: dict[str, Any] | None,
) -> dict[str, Any]:
    latest = latest_lottery_payload(
        lottery,
        prediction_root=prediction_root,
        settlement_root=settlement_root,
    )
    return {
        "latest_official_draw": latest,
        "pending_prediction": next_prediction_payload(
            lottery,
            prediction_root=prediction_root,
        )["pending_prediction"],
        "latest_settlement_summary": _latest_settlement_summary(settlement_root, lottery),
        "next_scheduled_action": None if automation is None else automation["next_action"],
        "next_run_at": None if automation is None else automation["next_run_at"],
    }


def _load_history(lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
    path = canonical_history_path(lottery)
    if not path.exists():
        return ()
    return load_draws_csv(path, lottery)


def _history_row(
    draw: HistoricalDraw,
    *,
    prediction_root: str | Path,
    settlement_root: str | Path,
) -> dict[str, Any]:
    settlement = _load_settlement_if_exists(settlement_root, draw.lottery, draw.draw_number)
    prediction = _load_prediction_if_exists(prediction_root, draw.lottery, draw.draw_number)
    financial = None if settlement is None else _settlement_financial(settlement)
    return {
        "draw_number": draw.draw_number,
        "draw_date": draw.draw_date.isoformat(),
        "main_numbers": draw.main_numbers,
        "bonus_numbers": draw.bonus_numbers,
        "prediction_available": prediction is not None,
        "best_match": None
        if prediction is None or prediction.evaluation is None
        else prediction.evaluation["best_match_count"],
        "paper_cost_yen": None if financial is None else financial["paper_total_cost_yen"],
        "paper_gross_winnings_yen": None
        if financial is None
        else financial["paper_gross_winnings_yen"],
        "paper_net_yen": None if financial is None else financial["paper_net_yen"],
        "settlement_status": None if settlement is None else settlement.financial_status,
    }


def _prediction_summary(record: PredictionRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "status": record.status,
        "target_draw_number": record.target_draw_number,
        "target_draw_date": record.target_draw_date,
        "generated_at": record.generated_at,
        "ticket_count": record.tickets_per_draw,
        "tickets": tuple(
            {"ticket_index": ticket.ticket_index, "numbers": ticket.numbers}
            for ticket in record.tickets
        ),
        "strategy": record.strategy,
        "model": record.model,
        "feature_version": record.feature_version,
        "feature_group": record.feature_group,
        "portfolio_method": record.portfolio_method,
        "seed": record.seed,
    }


def _settlement_ticket_results(settlement: DrawSettlement) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "ticket_index": ticket.ticket_index,
            "numbers": ticket.numbers,
            "main_matches": ticket.main_match_count,
            "bonus_matches": ticket.bonus_match_count,
            "prize_tier": ticket.prize_tier,
            "payout_yen": ticket.payout_yen,
        }
        for ticket in settlement.tickets
    )


def _settlement_financial(settlement: DrawSettlement) -> dict[str, Any]:
    return {
        "paper_total_cost_yen": settlement.paper_total_cost_yen,
        "paper_gross_winnings_yen": settlement.paper_gross_winnings_yen,
        "paper_net_yen": settlement.paper_net_yen,
        "financial_status": settlement.financial_status,
    }


def _load_settlement_if_exists(
    root: str | Path,
    lottery: LotteryDefinition,
    draw_number: int,
) -> DrawSettlement | None:
    path = settlement_path(root, lottery, draw_number)
    if not path.exists():
        return None
    return load_settlement(path)


def _load_prediction_if_exists(
    root: str | Path,
    lottery: LotteryDefinition,
    draw_number: int,
) -> PredictionRecord | None:
    path = prediction_record_path(root, lottery, draw_number)
    if not path.exists():
        return None
    return load_prediction_record(path)


def _latest_pending_prediction(
    root: str | Path,
    lottery: LotteryDefinition,
) -> PredictionRecord | None:
    directory = prediction_lottery_dir(root, lottery)
    if not directory.exists():
        return None
    pending = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "ledger.json":
            continue
        record = load_prediction_record(path)
        if record.status == PREDICTION_STATUS_PENDING:
            pending.append(record)
    return pending[0] if pending else None


def _latest_settlement_summary(
    root: str | Path,
    lottery: LotteryDefinition,
) -> dict[str, Any] | None:
    directory = settlement_lottery_dir(root, lottery)
    if not directory.exists():
        return None
    settlements = tuple(load_settlement(path) for path in sorted(directory.glob("*.json")))
    if not settlements:
        return None
    latest = settlements[-1]
    return {
        "draw_number": latest.draw_number,
        "draw_date": latest.draw_date,
        **_settlement_financial(latest),
    }


def _status_warnings(
    automation: dict[str, Any],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for code, payload in automation["lotteries"].items():
        if payload.get("warnings"):
            warnings.extend(f"{code}: {warning}" for warning in payload["warnings"])
    if automation.get("financial_pending_count"):
        warnings.append(f"payout pending: {automation['financial_pending_count']}")
    notification = notification_status()
    if notification["failed"]:
        warnings.append(f"notification failures: {notification['failed']}")
    return tuple(warnings)


def _public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_payload(item)
            for key, item in value.items()
            if key not in {"path", "record_path", "output_path", "prediction_record_path"}
            and not str(key).endswith("_path")
        }
    if isinstance(value, list | tuple):
        return tuple(_public_payload(item) for item in value)
    return value
