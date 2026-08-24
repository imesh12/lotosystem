from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from backend.app.domain.rules import get_lottery_definition
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.operations import (
    financial_summary_payload,
    history_payload,
    latest_lottery_payload,
    lotteries_payload,
    next_prediction_payload,
    notification_status_payload,
    settings_api_payload,
    status_payload,
)
from backend.app.research.settings import (
    SETTINGS_PATH,
    load_settings,
    settings_payload,
    update_settings,
)

router = APIRouter(tags=["operations"])
API_SETTINGS_PATH = SETTINGS_PATH


class LotterySettingsRequest(BaseModel):
    enabled: bool | None = None
    tickets_per_draw: int | None = None

    model_config = ConfigDict(extra="forbid")


class SettingsUpdateRequest(BaseModel):
    LOTO6: LotterySettingsRequest | None = None
    MINI_LOTO: LotterySettingsRequest | None = None
    email_enabled: bool | None = None

    model_config = ConfigDict(extra="forbid")


@router.get("/status")
def get_status() -> dict[str, Any]:
    return _safe(lambda: status_payload(settings=load_settings(API_SETTINGS_PATH)))


@router.get("/settings")
def get_settings_payload() -> dict[str, Any]:
    return settings_api_payload(load_settings(API_SETTINGS_PATH))


@router.put("/settings")
def put_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    updates = payload.model_dump(exclude_unset=True)
    try:
        return settings_payload(update_settings(updates, path=API_SETTINGS_PATH))
    except ResearchValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/lotteries")
def get_lotteries() -> dict[str, Any]:
    return lotteries_payload(load_settings(API_SETTINGS_PATH))


@router.get("/lotteries/{lottery}/latest")
def get_latest_lottery(lottery: str) -> dict[str, Any]:
    return _safe(lambda: latest_lottery_payload(_lottery(lottery)))


@router.get("/lotteries/{lottery}/next-prediction")
def get_next_prediction(lottery: str) -> dict[str, Any]:
    return _safe(lambda: next_prediction_payload(_lottery(lottery)))


@router.get("/lotteries/{lottery}/history")
def get_history(
    lottery: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return _safe(lambda: history_payload(_lottery(lottery), limit=limit, offset=offset))


@router.get("/financial/summary")
def get_financial_summary(
    lottery: str = Query(default="ALL"),
    period: str = Query(default="all_time"),
) -> dict[str, Any]:
    return _safe(lambda: financial_summary_payload(lottery_code=lottery, period=period))


@router.get("/notifications/status")
def get_notification_status() -> dict[str, Any]:
    return notification_status_payload()


def _lottery(code: str):
    try:
        return get_lottery_definition(code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"unknown lottery: {code}") from exc


def _safe(factory):
    try:
        return factory() if callable(factory) else factory
    except ResearchValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
