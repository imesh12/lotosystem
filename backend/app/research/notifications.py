from __future__ import annotations

import json
import os
import smtplib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import get_lottery_definition
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.production import (
    PREDICTION_ROOT,
    PREDICTION_STATUS_PENDING,
    PredictionRecord,
    load_prediction_record,
    prediction_lottery_dir,
)
from backend.app.research.settlement import (
    FINANCIAL_STATUS_COMPLETE,
    DrawSettlement,
    load_settlement,
)

NOTIFICATION_ROOT = Path("data") / "notifications"
NOTIFICATION_SCHEMA_VERSION = "stage14-notification-v1"

NOTIFICATION_DRAW_PROCESSED = "DRAW_PROCESSED"
NOTIFICATION_SOURCE_FAILURE = "SOURCE_FAILURE"
NOTIFICATION_PAYOUT_COMPLETED = "PAYOUT_COMPLETED"

DELIVERY_PENDING = "PENDING"
DELIVERY_SENT = "SENT"
DELIVERY_FAILED = "FAILED"
DELIVERY_DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class EmailConfig:
    enabled: bool
    from_address: str | None
    to_address: str | None
    smtp_host: str | None
    smtp_port: int
    username: str | None
    password: str | None
    use_tls: bool


class EmailEnvironmentSettings(BaseSettings):
    email_enabled: bool = Field(default=False, validation_alias="LOTO_EMAIL_ENABLED")
    email_from: str | None = Field(default=None, validation_alias="LOTO_EMAIL_FROM")
    email_to: str | None = Field(default=None, validation_alias="LOTO_EMAIL_TO")
    smtp_host: str | None = Field(default=None, validation_alias="LOTO_SMTP_HOST")
    smtp_port: int = Field(default=587, validation_alias="LOTO_SMTP_PORT")
    smtp_username: str | None = Field(default=None, validation_alias="LOTO_SMTP_USERNAME")
    smtp_password: str | None = Field(default=None, validation_alias="LOTO_SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, validation_alias="LOTO_SMTP_USE_TLS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@dataclass(frozen=True, slots=True)
class EmailPayload:
    subject: str
    body: str


class EmailSender(Protocol):
    def send(self, payload: EmailPayload) -> None: ...


@dataclass(frozen=True, slots=True)
class SMTPEmailSender:
    config: EmailConfig

    def send(self, payload: EmailPayload) -> None:
        _validate_email_config(self.config)
        message = EmailMessage()
        message["From"] = self.config.from_address or ""
        message["To"] = self.config.to_address or ""
        message["Subject"] = payload.subject
        message.set_content(payload.body)
        with smtplib.SMTP(self.config.smtp_host or "", self.config.smtp_port, timeout=20) as smtp:
            if self.config.use_tls:
                smtp.starttls()
            if self.config.username:
                smtp.login(self.config.username, self.config.password or "")
            smtp.send_message(message)


@dataclass(frozen=True, slots=True)
class NotificationRecord:
    schema_version: str
    notification_id: str
    notification_type: str
    lottery: str
    draw_number: int | None
    created_at: str
    subject: str
    body: str
    delivery_status: str
    attempt_count: int
    last_error: str | None
    sent_at: str | None
    source: dict[str, Any]


def email_config_from_env(env: dict[str, str] | None = None) -> EmailConfig:
    if env is None:
        settings = EmailEnvironmentSettings()
        return EmailConfig(
            enabled=settings.email_enabled,
            from_address=_empty_to_none(settings.email_from),
            to_address=_empty_to_none(settings.email_to),
            smtp_host=_empty_to_none(settings.smtp_host),
            smtp_port=settings.smtp_port,
            username=_empty_to_none(settings.smtp_username),
            password=_empty_to_none(settings.smtp_password),
            use_tls=settings.smtp_use_tls,
        )
    values = env
    return EmailConfig(
        enabled=_truthy(values.get("LOTO_EMAIL_ENABLED")),
        from_address=_empty_to_none(values.get("LOTO_EMAIL_FROM")),
        to_address=_empty_to_none(values.get("LOTO_EMAIL_TO")),
        smtp_host=_empty_to_none(values.get("LOTO_SMTP_HOST")),
        smtp_port=int(values.get("LOTO_SMTP_PORT") or "587"),
        username=_empty_to_none(values.get("LOTO_SMTP_USERNAME")),
        password=_empty_to_none(values.get("LOTO_SMTP_PASSWORD")),
        use_tls=_truthy(values.get("LOTO_SMTP_USE_TLS", "true")),
    )


def notification_status(
    *,
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
) -> dict[str, Any]:
    email_config = config or email_config_from_env()
    records = _load_all_notifications(notification_root)
    latest = records[-1] if records else None
    return {
        "enabled": email_config.enabled,
        "pending": sum(record.delivery_status == DELIVERY_PENDING for record in records),
        "failed": sum(record.delivery_status == DELIVERY_FAILED for record in records),
        "sent": sum(record.delivery_status == DELIVERY_SENT for record in records),
        "disabled": sum(record.delivery_status == DELIVERY_DISABLED for record in records),
        "latest_notification": None if latest is None else to_jsonable(latest),
    }


def send_pending_notifications(
    *,
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
    sender: EmailSender | None = None,
) -> dict[str, Any]:
    email_config = config or email_config_from_env()
    records = tuple(
        record
        for record in _load_all_notifications(notification_root)
        if record.delivery_status in {DELIVERY_PENDING, DELIVERY_FAILED, DELIVERY_DISABLED}
    )
    sent: list[str] = []
    failed: list[str] = []
    disabled: list[str] = []
    for record in records:
        delivered = _attempt_delivery(
            record,
            notification_root=notification_root,
            config=email_config,
            sender=sender,
        )
        if delivered.delivery_status == DELIVERY_SENT:
            sent.append(delivered.notification_id)
        elif delivered.delivery_status == DELIVERY_DISABLED:
            disabled.append(delivered.notification_id)
        else:
            failed.append(delivered.notification_id)
    return {"sent": tuple(sent), "failed": tuple(failed), "disabled": tuple(disabled)}


def send_test_email(
    *,
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
    sender: EmailSender | None = None,
) -> NotificationRecord:
    now = datetime.now(UTC)
    record = NotificationRecord(
        schema_version=NOTIFICATION_SCHEMA_VERSION,
        notification_id=f"TEST-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
        notification_type="TEST",
        lottery="ALL",
        draw_number=None,
        created_at=now.isoformat(),
        subject="[LotoSystem] Test Email",
        body="PAPER TRADING / SIMULATED\n\nThis is a LotoSystem email delivery test.",
        delivery_status=DELIVERY_PENDING,
        attempt_count=0,
        last_error=None,
        sent_at=None,
        source={"kind": "test"},
    )
    saved = save_notification(record, notification_path(notification_root, record.notification_id))
    return _attempt_delivery(
        saved,
        notification_root=notification_root,
        config=config or email_config_from_env(),
        sender=sender,
    )


def notify_draw_processed(
    settlement_path: str | Path,
    *,
    next_prediction_path: str | Path | None = None,
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
    sender: EmailSender | None = None,
    source_summary: dict[str, Any] | None = None,
) -> NotificationRecord:
    settlement = load_settlement(settlement_path)
    lottery = get_lottery_definition(settlement.lottery)
    prediction = load_prediction_record(settlement.prediction_record_path)
    next_prediction = (
        load_prediction_record(next_prediction_path)
        if next_prediction_path is not None and Path(next_prediction_path).exists()
        else _latest_pending_prediction(PREDICTION_ROOT, lottery)
    )
    report = build_draw_report(
        lottery,
        settlement,
        prediction,
        next_prediction,
        source_summary=source_summary or {},
    )
    payload = render_draw_processed_email(report)
    notification_id = f"{NOTIFICATION_DRAW_PROCESSED}-{settlement.lottery}-{settlement.draw_number}"
    record = _new_or_existing_notification(
        notification_root,
        notification_id=notification_id,
        notification_type=NOTIFICATION_DRAW_PROCESSED,
        lottery=settlement.lottery,
        draw_number=settlement.draw_number,
        payload=payload,
        source={"settlement_path": str(settlement_path), "source_summary": source_summary or {}},
    )
    return _attempt_delivery(
        record,
        notification_root=notification_root,
        config=config or email_config_from_env(),
        sender=sender,
    )


def notify_source_failure(
    lottery: LotteryDefinition,
    *,
    current_time: datetime,
    latest_history: dict[str, Any],
    pending_prediction: dict[str, Any] | None,
    sources_attempted: tuple[dict[str, Any], ...],
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
    sender: EmailSender | None = None,
) -> NotificationRecord:
    day = current_time.date().isoformat().replace("-", "")
    draw_number = (
        None if pending_prediction is None else int(pending_prediction["target_draw_number"])
    )
    notification_id = f"{NOTIFICATION_SOURCE_FAILURE}-{lottery.code}-{draw_number or 'NONE'}-{day}"
    payload = render_source_failure_email(
        lottery,
        current_time=current_time,
        latest_history=latest_history,
        pending_prediction=pending_prediction,
        sources_attempted=sources_attempted,
    )
    record = _new_or_existing_notification(
        notification_root,
        notification_id=notification_id,
        notification_type=NOTIFICATION_SOURCE_FAILURE,
        lottery=str(lottery.code),
        draw_number=draw_number,
        payload=payload,
        source={
            "latest_history": latest_history,
            "pending_prediction": pending_prediction,
            "sources_attempted": sources_attempted,
        },
    )
    return _attempt_delivery(
        record,
        notification_root=notification_root,
        config=config or email_config_from_env(),
        sender=sender,
    )


def notify_payout_completed(
    settlement_path: str | Path,
    *,
    notification_root: str | Path = NOTIFICATION_ROOT,
    config: EmailConfig | None = None,
    sender: EmailSender | None = None,
) -> NotificationRecord:
    settlement = load_settlement(settlement_path)
    if settlement.financial_status != FINANCIAL_STATUS_COMPLETE:
        raise ResearchValidationError("payout-completed notification requires COMPLETE settlement")
    payload = render_payout_completed_email(settlement)
    notification_id = (
        f"{NOTIFICATION_PAYOUT_COMPLETED}-{settlement.lottery}-{settlement.draw_number}"
    )
    record = _new_or_existing_notification(
        notification_root,
        notification_id=notification_id,
        notification_type=NOTIFICATION_PAYOUT_COMPLETED,
        lottery=settlement.lottery,
        draw_number=settlement.draw_number,
        payload=payload,
        source={"settlement_path": str(settlement_path)},
    )
    return _attempt_delivery(
        record,
        notification_root=notification_root,
        config=config or email_config_from_env(),
        sender=sender,
    )


def build_draw_report(
    lottery: LotteryDefinition,
    settlement: DrawSettlement,
    prediction: PredictionRecord,
    next_prediction: PredictionRecord | None,
    *,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage14-draw-report-v1",
        "paper_trading": True,
        "lottery": str(lottery.code),
        "draw_number": settlement.draw_number,
        "draw_date": settlement.draw_date,
        "official_result": {
            "main_numbers": settlement.actual_main_numbers,
            "bonus_numbers": settlement.actual_bonus_numbers,
        },
        "prediction": {
            "prediction_id": Path(settlement.prediction_record_path).stem,
            "generated_at": prediction.generated_at,
            "tickets": tuple(
                {"ticket_index": ticket.ticket_index, "numbers": ticket.numbers}
                for ticket in prediction.tickets
            ),
        },
        "ticket_results": tuple(
            {
                "ticket_index": ticket.ticket_index,
                "numbers": ticket.numbers,
                "main_matches": ticket.main_match_count,
                "bonus_matches": ticket.bonus_match_count,
                "prize_tier": ticket.prize_tier,
                "payout_yen": ticket.payout_yen,
            }
            for ticket in settlement.tickets
        ),
        "paper_financial": {
            "ticket_count": settlement.ticket_count,
            "cost_yen": settlement.paper_total_cost_yen,
            "gross_winnings_yen": settlement.paper_gross_winnings_yen,
            "net_yen": settlement.paper_net_yen,
            "financial_status": settlement.financial_status,
        },
        "next_prediction": None
        if next_prediction is None
        else {
            "draw_number": next_prediction.target_draw_number,
            "expected_date": next_prediction.target_draw_date,
            "tickets": tuple(
                {"ticket_index": ticket.ticket_index, "numbers": ticket.numbers}
                for ticket in next_prediction.tickets
            ),
        },
        "source": source_summary,
    }


def render_draw_processed_email(report: dict[str, Any]) -> EmailPayload:
    lottery = report["lottery"]
    subject = f"[LotoSystem] {lottery} #{report['draw_number']} Result"
    lines = [
        "PAPER TRADING / SIMULATED",
        "",
        f"{lottery} #{report['draw_number']}",
        str(report["draw_date"]),
        "",
        "Winning Numbers",
        _format_numbers(report["official_result"]["main_numbers"]),
        f"Bonus: {_format_numbers(report['official_result']['bonus_numbers'])}",
        "",
        "Our Prediction",
    ]
    for ticket in report["ticket_results"]:
        lines.append("")
        lines.append(f"Set {ticket['ticket_index']}: {_format_numbers(ticket['numbers'])}")
        lines.append(f"Result: {ticket['main_matches']} matches")
        if ticket["prize_tier"] == "NO_PRIZE":
            lines.append("Prize: No prize")
        else:
            lines.append(f"Prize: {ticket['prize_tier']}")
            lines.append(f"Paper payout: {_format_yen(ticket['payout_yen'])}")
    financial = report["paper_financial"]
    lines.extend(
        [
            "",
            "Paper Trading",
            f"Cost: {_format_yen(financial['cost_yen'])}",
            f"Winnings: {_format_yen(financial['gross_winnings_yen'])}",
            f"Net: {_format_yen(financial['net_yen'], signed=True)}",
            f"Financial status: {financial['financial_status']}",
        ]
    )
    next_prediction = report["next_prediction"]
    if next_prediction is not None:
        lines.extend(["", "Next Prediction", f"{lottery} #{next_prediction['draw_number']}"])
        for ticket in next_prediction["tickets"]:
            lines.append(f"Set {ticket['ticket_index']}: {_format_numbers(ticket['numbers'])}")
    return EmailPayload(subject=subject, body="\n".join(lines))


def render_source_failure_email(
    lottery: LotteryDefinition,
    *,
    current_time: datetime,
    latest_history: dict[str, Any],
    pending_prediction: dict[str, Any] | None,
    sources_attempted: tuple[dict[str, Any], ...],
) -> EmailPayload:
    subject = f"[LotoSystem] {lottery.code} Source Failure"
    lines = [
        "PAPER TRADING / SIMULATED",
        "",
        f"{lottery.code} source failure",
        f"Time: {current_time.isoformat()}",
        f"Latest canonical draw: {latest_history.get('latest_draw_number')}",
        f"Latest canonical date: {latest_history.get('latest_draw_date')}",
        "Pending prediction: "
        f"{None if pending_prediction is None else pending_prediction.get('draw')}",
        "",
        "Sources attempted",
    ]
    for attempt in sources_attempted:
        lines.append(
            f"- {attempt.get('source')}: {attempt.get('result')} {attempt.get('error') or ''}"
        )
    lines.extend(["", "Manual action is available with add-result after official verification."])
    return EmailPayload(subject=subject, body="\n".join(lines))


def render_payout_completed_email(settlement: DrawSettlement) -> EmailPayload:
    subject = f"[LotoSystem] {settlement.lottery} #{settlement.draw_number} Payout Completed"
    body = "\n".join(
        [
            "PAPER TRADING / SIMULATED",
            "",
            f"{settlement.lottery} #{settlement.draw_number} payout completed",
            f"Cost: {_format_yen(settlement.paper_total_cost_yen)}",
            f"Winnings: {_format_yen(settlement.paper_gross_winnings_yen)}",
            f"Net: {_format_yen(settlement.paper_net_yen, signed=True)}",
        ]
    )
    return EmailPayload(subject=subject, body=body)


def save_notification(record: NotificationRecord, path: str | Path) -> NotificationRecord:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(research_result_json(record), encoding="utf-8")
    return record


def load_notification(path: str | Path) -> NotificationRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NotificationRecord(
        schema_version=payload["schema_version"],
        notification_id=payload["notification_id"],
        notification_type=payload["notification_type"],
        lottery=payload["lottery"],
        draw_number=payload["draw_number"],
        created_at=payload["created_at"],
        subject=payload["subject"],
        body=payload["body"],
        delivery_status=payload["delivery_status"],
        attempt_count=int(payload["attempt_count"]),
        last_error=payload["last_error"],
        sent_at=payload["sent_at"],
        source=payload["source"],
    )


def notification_path(root: str | Path, notification_id: str) -> Path:
    return Path(root) / f"{notification_id}.json"


def _new_or_existing_notification(
    root: str | Path,
    *,
    notification_id: str,
    notification_type: str,
    lottery: str,
    draw_number: int | None,
    payload: EmailPayload,
    source: dict[str, Any],
) -> NotificationRecord:
    path = notification_path(root, notification_id)
    if path.exists():
        return load_notification(path)
    record = NotificationRecord(
        schema_version=NOTIFICATION_SCHEMA_VERSION,
        notification_id=notification_id,
        notification_type=notification_type,
        lottery=lottery,
        draw_number=draw_number,
        created_at=datetime.now(UTC).isoformat(),
        subject=payload.subject,
        body=payload.body,
        delivery_status=DELIVERY_PENDING,
        attempt_count=0,
        last_error=None,
        sent_at=None,
        source=source,
    )
    return save_notification(record, path)


def _attempt_delivery(
    record: NotificationRecord,
    *,
    notification_root: str | Path,
    config: EmailConfig,
    sender: EmailSender | None,
) -> NotificationRecord:
    if record.delivery_status == DELIVERY_SENT:
        return record
    if not config.enabled:
        updated = replace(
            record,
            delivery_status=DELIVERY_DISABLED,
            last_error=None,
        )
        save_notification(updated, notification_path(notification_root, record.notification_id))
        return updated
    try:
        (sender or SMTPEmailSender(config)).send(EmailPayload(record.subject, record.body))
    except Exception as exc:
        updated = replace(
            record,
            delivery_status=DELIVERY_FAILED,
            attempt_count=record.attempt_count + 1,
            last_error=_sanitize_error(str(exc), config),
        )
        save_notification(updated, notification_path(notification_root, record.notification_id))
        return updated
    updated = replace(
        record,
        delivery_status=DELIVERY_SENT,
        attempt_count=record.attempt_count + 1,
        last_error=None,
        sent_at=datetime.now(UTC).isoformat(),
    )
    save_notification(updated, notification_path(notification_root, record.notification_id))
    return updated


def _latest_pending_prediction(
    prediction_root: str | Path,
    lottery: LotteryDefinition,
) -> PredictionRecord | None:
    directory = prediction_lottery_dir(prediction_root, lottery)
    if not directory.exists():
        return None
    pending: list[PredictionRecord] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "ledger.json":
            continue
        record = load_prediction_record(path)
        if record.status == PREDICTION_STATUS_PENDING:
            pending.append(record)
    return pending[-1] if pending else None


def _load_all_notifications(root: str | Path) -> tuple[NotificationRecord, ...]:
    directory = Path(root)
    if not directory.exists():
        return ()
    return tuple(load_notification(path) for path in sorted(directory.glob("*.json")))


def _validate_email_config(config: EmailConfig) -> None:
    missing = []
    if not config.from_address:
        missing.append("LOTO_EMAIL_FROM")
    if not config.to_address:
        missing.append("LOTO_EMAIL_TO")
    if not config.smtp_host:
        missing.append("LOTO_SMTP_HOST")
    if missing:
        raise ResearchValidationError(
            "email enabled but configuration is missing: " + ", ".join(missing)
        )


def _format_numbers(numbers: tuple[int, ...] | list[int]) -> str:
    return " ".join(f"{number:02d}" for number in numbers)


def _format_yen(value: int | None, *, signed: bool = False) -> str:
    if value is None:
        return "pending"
    if signed and value < 0:
        return f"-¥{abs(value):,}"
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}¥{value:,}"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _sanitize_error(message: str, config: EmailConfig) -> str:
    sanitized = message
    for secret in (config.password, os.environ.get("LOTO_SMTP_PASSWORD")):
        if secret:
            sanitized = sanitized.replace(secret, "[redacted]")
    return sanitized
