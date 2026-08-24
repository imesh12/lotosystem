from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.automation import ACTION_RESULT_PROCESSED, run_automation_once
from backend.app.research.data import HistoricalDraw
from backend.app.research.history_import import (
    HISTORY_UPDATE_NEW_RESULT,
    write_canonical_history_csv,
)
from backend.app.research.notifications import (
    DELIVERY_DISABLED,
    DELIVERY_FAILED,
    DELIVERY_SENT,
    EmailConfig,
    EmailPayload,
    build_draw_report,
    notification_status,
    notify_draw_processed,
    notify_payout_completed,
    notify_source_failure,
    render_draw_processed_email,
    send_pending_notifications,
)
from backend.app.research.operational_cycle import (
    CycleHistorySummary,
    CycleNextPredictionSummary,
    OperationalCycleResult,
)
from backend.app.research.payouts import manual_draw_payout
from backend.app.research.prize import match_ticket
from backend.app.research.production import (
    PredictionRecord,
    PredictionTicket,
    save_prediction_record,
)
from backend.app.research.settlement import (
    FINANCIAL_STATUS_COMPLETE,
    FINANCIAL_STATUS_PAYOUT_PENDING,
    build_settlement,
    load_settlement,
    save_settlement,
    settlement_path,
)


@dataclass
class FakeSender:
    sent: list[EmailPayload]
    error: str | None = None

    def send(self, payload: EmailPayload) -> None:
        if self.error:
            raise RuntimeError(self.error)
        self.sent.append(payload)


def _email_config(enabled: bool = True) -> EmailConfig:
    return EmailConfig(
        enabled=enabled,
        from_address="from@example.test",
        to_address="to@example.test",
        smtp_host="smtp.example.test",
        smtp_port=587,
        username="user",
        password="secret",
        use_tls=True,
    )


def _draw(lottery: LotteryDefinition) -> HistoricalDraw:
    if lottery.code == LOTO6.code:
        return HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    return HistoricalDraw(MINI_LOTO, 1401, date(2026, 8, 25), (1, 2, 3, 4, 5), (6,))


def _record_for_tickets(
    lottery: LotteryDefinition,
    draw: HistoricalDraw,
    tickets: tuple[tuple[int, ...], ...],
    *,
    status: str = "EVALUATED",
) -> PredictionRecord:
    ticket_records = tuple(
        PredictionTicket(index + 1, lottery.validate_main_numbers(ticket), 0.0)
        for index, ticket in enumerate(tickets)
    )
    ticket_results = []
    for ticket in ticket_records:
        match = match_ticket(ticket.numbers, draw, lottery)
        ticket_results.append(
            {
                "ticket_index": ticket.ticket_index,
                "numbers": ticket.numbers,
                "main_match_count": match.main_match_count,
                "bonus_match_count": match.bonus_match_count,
                "prize_category": match.prize_name,
                "qualifies_for_prize": match.qualifies_for_prize,
            }
        )
    return PredictionRecord(
        schema_version="stage10-production-prediction-v1",
        status=status,
        lottery=str(lottery.code),
        target_draw_number=draw.draw_number,
        target_draw_date=draw.draw_date.isoformat(),
        generated_at="2026-08-24T00:00:00+00:00",
        dataset_hash="dataset",
        latest_source_draw_number=draw.draw_number - 1,
        latest_source_draw_date="2026-08-01",
        strategy="test",
        model="test",
        model_parameters={},
        feature_version="test",
        feature_group="test",
        feature_names=(),
        portfolio_method="test",
        portfolio_version="test",
        seed=123456,
        tickets_per_draw=len(ticket_records),
        ticket_price_yen=lottery.ticket_price_yen,
        cost_yen=len(ticket_records) * lottery.ticket_price_yen,
        gross_winnings_yen=None,
        sklearn_version="test",
        config={},
        tickets=ticket_records,
        evaluation=None
        if status != "EVALUATED"
        else {
            "evaluated_at": "2026-08-24T01:00:00+00:00",
            "actual_draw_number": draw.draw_number,
            "actual_draw_date": draw.draw_date.isoformat(),
            "actual_main_numbers": draw.main_numbers,
            "actual_bonus_numbers": draw.bonus_numbers,
            "ticket_results": tuple(ticket_results),
            "best_match_count": max(result["main_match_count"] for result in ticket_results),
            "prize_qualified_ticket_count": sum(
                result["qualifies_for_prize"] for result in ticket_results
            ),
            "gross_winnings_yen": None,
        },
        warnings=(),
    )


def _saved_settlement(
    tmp_path: Path,
    lottery: LotteryDefinition,
    tickets: tuple[tuple[int, ...], ...],
    *,
    payouts: dict[str, int],
) -> tuple[Path, PredictionRecord]:
    draw = _draw(lottery)
    record = _record_for_tickets(lottery, draw, tickets)
    prediction_path = tmp_path / "predictions" / str(lottery.code) / f"{draw.draw_number}.json"
    save_prediction_record(record, prediction_path)
    payout_records = tuple(
        manual_draw_payout(lottery, draw_number=draw.draw_number, prize_tier=tier, payout_yen=value)
        for tier, value in payouts.items()
    )
    settlement = build_settlement(record, lottery, str(prediction_path), payout_records)
    path = settlement_path(tmp_path / "settlements", lottery, draw.draw_number)
    save_settlement(settlement, path)
    return path, record


def test_completed_loto6_report_and_rendering_includes_three_tickets(tmp_path: Path) -> None:
    path, record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((1, 2, 3, 8, 9, 10), (8, 9, 10, 11, 12, 13), (1, 2, 3, 4, 8, 9)),
        payouts={"5th": 1_000, "4th": 6_800},
    )

    report = build_draw_report(
        LOTO6,
        load_settlement(path),
        record,
        None,
        source_summary={"selected_source": "secondary"},
    )
    rendered = render_draw_processed_email(report)

    assert report["paper_trading"] is True
    assert len(report["ticket_results"]) == 3
    assert "PAPER TRADING / SIMULATED" in rendered.body
    assert "Prize: 5th" in rendered.body
    assert "Prize: No prize" in rendered.body
    assert "Net: +¥7,200" in rendered.body


def test_mini_loto_report_and_payout_pending(tmp_path: Path) -> None:
    path, record = _saved_settlement(
        tmp_path,
        MINI_LOTO,
        ((1, 2, 3, 7, 8), (7, 8, 9, 10, 11), (1, 2, 3, 4, 7)),
        payouts={"3rd": 10_000},
    )
    report = build_draw_report(
        MINI_LOTO,
        load_settlement(path),
        record,
        None,
        source_summary={},
    )

    assert report["lottery"] == "MINI_LOTO"
    assert report["paper_financial"]["financial_status"] == FINANCIAL_STATUS_PAYOUT_PENDING
    assert "Winnings: pending" in render_draw_processed_email(report).body


def test_negative_net_no_winning_tickets(tmp_path: Path) -> None:
    path, record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )
    report = build_draw_report(
        LOTO6,
        load_settlement(path),
        record,
        None,
        source_summary={},
    )

    assert report["paper_financial"]["financial_status"] == FINANCIAL_STATUS_COMPLETE
    assert "Net: -¥200" in render_draw_processed_email(report).body


def test_draw_processed_notification_idempotency_and_disabled_email(tmp_path: Path) -> None:
    path, _record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )
    sender = FakeSender([])

    first = notify_draw_processed(
        path,
        notification_root=tmp_path / "notifications",
        config=_email_config(enabled=False),
        sender=sender,
    )
    second = notify_draw_processed(
        path,
        notification_root=tmp_path / "notifications",
        config=_email_config(enabled=True),
        sender=sender,
    )

    assert first.delivery_status == DELIVERY_DISABLED
    assert second.notification_id == first.notification_id
    assert second.delivery_status == DELIVERY_SENT
    assert len(sender.sent) == 1


def test_source_failure_and_payout_completed_notifications(tmp_path: Path) -> None:
    sender = FakeSender([])
    source = notify_source_failure(
        LOTO6,
        current_time=datetime.fromisoformat("2026-08-24T21:30:00+09:00"),
        latest_history={"latest_draw_number": 2130, "latest_draw_date": "2026-08-20"},
        pending_prediction={"target_draw_number": 2131, "draw": 2131},
        sources_attempted=({"source": "mizuho", "result": "SOURCE_FAILURE", "error": "403"},),
        notification_root=tmp_path / "notifications",
        config=_email_config(),
        sender=sender,
    )
    settlement_path_value, _record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )
    payout = notify_payout_completed(
        settlement_path_value,
        notification_root=tmp_path / "notifications",
        config=_email_config(),
        sender=sender,
    )

    assert source.delivery_status == DELIVERY_SENT
    assert payout.delivery_status == DELIVERY_SENT
    assert len(sender.sent) == 2


def test_failed_notification_retry_and_secret_redaction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LOTO_SMTP_PASSWORD", "secret")
    path, _record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )
    failed = notify_draw_processed(
        path,
        notification_root=tmp_path / "notifications",
        config=_email_config(),
        sender=FakeSender([], error="bad secret"),
    )

    retried = send_pending_notifications(
        notification_root=tmp_path / "notifications",
        config=_email_config(),
        sender=FakeSender([]),
    )
    payload = json.loads(
        (tmp_path / "notifications" / f"{failed.notification_id}.json").read_text()
    )

    assert failed.delivery_status == DELIVERY_FAILED
    assert failed.last_error == "bad [redacted]"
    assert retried["sent"] == (failed.notification_id,)
    assert "secret" not in json.dumps(payload)


def test_notification_status_counts(tmp_path: Path) -> None:
    path, _record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )
    notify_draw_processed(
        path,
        notification_root=tmp_path / "notifications",
        config=_email_config(enabled=False),
        sender=FakeSender([]),
    )

    status = notification_status(notification_root=tmp_path / "notifications")

    assert status["enabled"] is False
    assert status["disabled"] == 1
    assert status["latest_notification"]["notification_type"] == "DRAW_PROCESSED"


def test_automation_smtp_failure_does_not_change_lifecycle_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    history = tmp_path / "loto6.csv"
    write_canonical_history_csv(
        (HistoricalDraw(LOTO6, 2130, date(2026, 8, 20), (8, 9, 10, 11, 12, 13), (1,)),),
        history,
    )
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", lambda _: history)
    settlement_path_value, _record = _saved_settlement(
        tmp_path,
        LOTO6,
        ((8, 9, 10, 11, 12, 13),),
        payouts={},
    )

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        return OperationalCycleResult(
            lottery="LOTO6",
            cycle_id="cycle",
            history=CycleHistorySummary(
                2130,
                2131,
                1,
                str(history),
                HISTORY_UPDATE_NEW_RESULT,
                "secondary",
                True,
                (),
            ),
            evaluated_predictions=(2131,),
            settlements=(str(settlement_path_value),),
            next_prediction=CycleNextPredictionSummary(
                2132,
                "2026-08-27",
                "PENDING",
                3,
                "next.json",
                True,
            ),
            cycle_record_path="cycle.json",
            errors=(),
            warnings=(),
        )

    payload = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        email_config=_email_config(),
        email_sender=FakeSender([], error="smtp down"),
        now=datetime.fromisoformat("2026-08-24T21:30:00+09:00"),
        cycle_runner=runner,
    )

    loto = payload["lotteries"][0]
    assert loto["action"] == ACTION_RESULT_PROCESSED
    assert loto["notifications"][0]["delivery_status"] == DELIVERY_FAILED
    assert "notification failed" not in " ".join(loto["warnings"])
