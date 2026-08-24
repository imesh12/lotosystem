from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import to_jsonable

SETTINGS_PATH = Path("config") / "operational_settings.json"
MIN_TICKETS_PER_DRAW = 1
MAX_TICKETS_PER_DRAW = 20


@dataclass(frozen=True, slots=True)
class LotteryOperationalSettings:
    enabled: bool = True
    tickets_per_draw: int = 3


@dataclass(frozen=True, slots=True)
class OperationalSettings:
    loto6: LotteryOperationalSettings
    mini_loto: LotteryOperationalSettings
    email_enabled: bool = False


def default_settings() -> OperationalSettings:
    return OperationalSettings(
        loto6=LotteryOperationalSettings(),
        mini_loto=LotteryOperationalSettings(),
        email_enabled=False,
    )


def load_settings(path: str | Path = SETTINGS_PATH) -> OperationalSettings:
    settings_path = Path(path)
    if not settings_path.exists():
        return default_settings()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    return _settings_from_payload(payload)


def save_settings(
    settings: OperationalSettings,
    path: str | Path = SETTINGS_PATH,
) -> OperationalSettings:
    _validate_settings(settings)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_jsonable(settings), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return settings


def update_settings(
    updates: dict[str, Any],
    *,
    path: str | Path = SETTINGS_PATH,
) -> OperationalSettings:
    _reject_unknown_fields(updates, {"LOTO6", "MINI_LOTO", "email_enabled"})
    current = load_settings(path)
    updated = current
    if "LOTO6" in updates:
        updated = replace(updated, loto6=_updated_lottery_settings(current.loto6, updates["LOTO6"]))
    if "MINI_LOTO" in updates:
        updated = replace(
            updated,
            mini_loto=_updated_lottery_settings(current.mini_loto, updates["MINI_LOTO"]),
        )
    if "email_enabled" in updates:
        if not isinstance(updates["email_enabled"], bool):
            raise ResearchValidationError("email_enabled must be a boolean")
        updated = replace(updated, email_enabled=updates["email_enabled"])
    return save_settings(updated, path)


def lottery_settings(
    lottery: LotteryDefinition,
    *,
    settings: OperationalSettings | None = None,
    path: str | Path = SETTINGS_PATH,
) -> LotteryOperationalSettings:
    active = settings or load_settings(path)
    if lottery.code == LOTO6.code:
        return active.loto6
    if lottery.code == MINI_LOTO.code:
        return active.mini_loto
    raise ResearchValidationError(f"unsupported lottery: {lottery.code}")


def effective_lottery_production_config(
    lottery: LotteryDefinition,
    *,
    settings: OperationalSettings | None = None,
    path: str | Path = SETTINGS_PATH,
) -> dict[str, Any]:
    selected = lottery_settings(lottery, settings=settings, path=path)
    return {
        "lottery": str(lottery.code),
        "enabled": selected.enabled,
        "tickets_per_draw": selected.tickets_per_draw,
    }


def settings_payload(settings: OperationalSettings) -> dict[str, Any]:
    return {
        "LOTO6": to_jsonable(settings.loto6),
        "MINI_LOTO": to_jsonable(settings.mini_loto),
        "email_enabled": settings.email_enabled,
    }


def _settings_from_payload(payload: dict[str, Any]) -> OperationalSettings:
    _reject_unknown_fields(payload, {"loto6", "mini_loto", "LOTO6", "MINI_LOTO", "email_enabled"})
    settings = OperationalSettings(
        loto6=_lottery_settings_from_payload(payload.get("loto6") or payload.get("LOTO6") or {}),
        mini_loto=_lottery_settings_from_payload(
            payload.get("mini_loto") or payload.get("MINI_LOTO") or {}
        ),
        email_enabled=bool(payload.get("email_enabled", False)),
    )
    _validate_settings(settings)
    return settings


def _lottery_settings_from_payload(payload: dict[str, Any]) -> LotteryOperationalSettings:
    _reject_unknown_fields(payload, {"enabled", "tickets_per_draw"})
    settings = LotteryOperationalSettings(
        enabled=bool(payload.get("enabled", True)),
        tickets_per_draw=int(payload.get("tickets_per_draw", 3)),
    )
    _validate_lottery_settings(settings)
    return settings


def _updated_lottery_settings(
    current: LotteryOperationalSettings,
    updates: dict[str, Any],
) -> LotteryOperationalSettings:
    if not isinstance(updates, dict):
        raise ResearchValidationError("lottery settings update must be an object")
    _reject_unknown_fields(updates, {"enabled", "tickets_per_draw"})
    updated = current
    if "enabled" in updates:
        if not isinstance(updates["enabled"], bool):
            raise ResearchValidationError("enabled must be a boolean")
        updated = replace(updated, enabled=updates["enabled"])
    if "tickets_per_draw" in updates:
        if not isinstance(updates["tickets_per_draw"], int):
            raise ResearchValidationError("tickets_per_draw must be an integer")
        updated = replace(updated, tickets_per_draw=updates["tickets_per_draw"])
    _validate_lottery_settings(updated)
    return updated


def _validate_settings(settings: OperationalSettings) -> None:
    _validate_lottery_settings(settings.loto6)
    _validate_lottery_settings(settings.mini_loto)


def _validate_lottery_settings(settings: LotteryOperationalSettings) -> None:
    if settings.tickets_per_draw < MIN_TICKETS_PER_DRAW:
        raise ResearchValidationError("tickets_per_draw must be positive")
    if settings.tickets_per_draw > MAX_TICKETS_PER_DRAW:
        raise ResearchValidationError(f"tickets_per_draw must be <= {MAX_TICKETS_PER_DRAW}")


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ResearchValidationError(f"unknown settings field(s): {', '.join(unknown)}")
