from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.payouts import (
    DrawPayout,
    collect_smbc_draw_payouts,
    manual_draw_payout,
    merge_draw_payouts,
)
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.production import (
    PREDICTION_ROOT,
    PREDICTION_STATUS_EVALUATED,
    PredictionRecord,
    load_prediction_record,
    prediction_lottery_dir,
)

SETTLEMENT_SCHEMA_VERSION = "stage12-paper-settlement-v1"
SETTLEMENT_ROOT = Path("data") / "settlements"
FINANCIAL_STATUS_COMPLETE = "COMPLETE"
FINANCIAL_STATUS_PAYOUT_PENDING = "PAYOUT_PENDING"
NO_PRIZE = "NO_PRIZE"


@dataclass(frozen=True, slots=True)
class TicketSettlement:
    ticket_index: int
    numbers: tuple[int, ...]
    main_match_count: int
    bonus_match_count: int
    prize_tier: str
    payout_yen: int | None


@dataclass(frozen=True, slots=True)
class DrawSettlement:
    schema_version: str
    lottery: str
    draw_number: int
    draw_date: str
    prediction_record_path: str
    prediction_generated_at: str
    prediction_dataset_hash: str
    settled_at: str
    actual_main_numbers: tuple[int, ...]
    actual_bonus_numbers: tuple[int, ...]
    payouts: tuple[DrawPayout, ...]
    tickets: tuple[TicketSettlement, ...]
    ticket_count: int
    ticket_price_yen: int
    paper_total_cost_yen: int
    paper_gross_winnings_yen: int | None
    paper_net_yen: int | None
    financial_status: str
    warnings: tuple[str, ...]


def settlement_lottery_dir(root: str | Path, lottery: LotteryDefinition) -> Path:
    return Path(root) / str(lottery.code)


def settlement_path(root: str | Path, lottery: LotteryDefinition, draw_number: int) -> Path:
    return settlement_lottery_dir(root, lottery) / f"{draw_number}.json"


def settle_evaluated_predictions(
    lottery: LotteryDefinition,
    *,
    prediction_root: str | Path = PREDICTION_ROOT,
    settlement_root: str | Path = SETTLEMENT_ROOT,
) -> tuple[str, ...]:
    paths: list[str] = []
    for path in sorted(prediction_lottery_dir(prediction_root, lottery).glob("*.json")):
        if path.name == "ledger.json":
            continue
        record = load_prediction_record(path)
        if record.status != PREDICTION_STATUS_EVALUATED:
            continue
        destination = settlement_path(settlement_root, lottery, record.target_draw_number)
        if destination.exists():
            existing = load_settlement(destination)
            if existing.financial_status == FINANCIAL_STATUS_COMPLETE:
                continue
        payouts = _try_collect_payouts(lottery, record.target_draw_number)
        settlement = build_settlement(record, lottery, str(path), payouts)
        saved = save_settlement(
            settlement,
            destination,
        )
        paths.append(str(saved))
    write_financial_ledger(settlement_root)
    return tuple(paths)


def build_settlement(
    record: PredictionRecord,
    lottery: LotteryDefinition,
    prediction_record_path: str,
    payouts: tuple[DrawPayout, ...],
    *,
    settled_at: datetime | None = None,
) -> DrawSettlement:
    if record.status != PREDICTION_STATUS_EVALUATED or record.evaluation is None:
        raise ResearchValidationError("prediction must be evaluated before settlement")
    merged_payouts = merge_draw_payouts((), payouts, lottery)
    payout_by_tier = {payout.prize_tier: payout for payout in merged_payouts}
    tickets: list[TicketSettlement] = []
    payout_pending = False
    for result in record.evaluation["ticket_results"]:
        tier = result["prize_category"] or NO_PRIZE
        payout_yen: int | None = 0
        if tier != NO_PRIZE:
            payout = payout_by_tier.get(tier)
            if payout is None:
                payout_yen = None
                payout_pending = True
            else:
                payout_yen = payout.payout_yen
        tickets.append(
            TicketSettlement(
                ticket_index=int(result["ticket_index"]),
                numbers=tuple(result["numbers"]),
                main_match_count=int(result["main_match_count"]),
                bonus_match_count=int(result["bonus_match_count"]),
                prize_tier=tier,
                payout_yen=payout_yen,
            )
        )

    cost = record.tickets_per_draw * record.ticket_price_yen
    gross = None if payout_pending else sum(ticket.payout_yen or 0 for ticket in tickets)
    net = None if gross is None else gross - cost
    return DrawSettlement(
        schema_version=SETTLEMENT_SCHEMA_VERSION,
        lottery=str(lottery.code),
        draw_number=record.target_draw_number,
        draw_date=record.target_draw_date,
        prediction_record_path=prediction_record_path,
        prediction_generated_at=record.generated_at,
        prediction_dataset_hash=record.dataset_hash,
        settled_at=(settled_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        actual_main_numbers=tuple(record.evaluation["actual_main_numbers"]),
        actual_bonus_numbers=tuple(record.evaluation["actual_bonus_numbers"]),
        payouts=merged_payouts,
        tickets=tuple(tickets),
        ticket_count=record.tickets_per_draw,
        ticket_price_yen=record.ticket_price_yen,
        paper_total_cost_yen=cost,
        paper_gross_winnings_yen=gross,
        paper_net_yen=net,
        financial_status=FINANCIAL_STATUS_PAYOUT_PENDING
        if payout_pending
        else FINANCIAL_STATUS_COMPLETE,
        warnings=(
            "Paper-trading settlement only; no ticket purchase is recorded.",
            "paper_gross_winnings_yen and paper_net_yen are simulated accounting fields.",
        ),
    )


def save_settlement(settlement: DrawSettlement, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_settlement(destination)
        if _settlement_scientific_payload(existing) == _settlement_scientific_payload(settlement):
            return destination
        if existing.financial_status == FINANCIAL_STATUS_PAYOUT_PENDING:
            destination.write_text(research_result_json(settlement), encoding="utf-8")
            return destination
        raise ResearchValidationError(
            f"conflicting completed settlement for {settlement.lottery} #{settlement.draw_number}"
        )
    destination.write_text(research_result_json(settlement), encoding="utf-8")
    return destination


def load_settlement(path: str | Path) -> DrawSettlement:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payouts = tuple(DrawPayout(**payout) for payout in payload["payouts"])
    tickets = tuple(TicketSettlement(**ticket) for ticket in payload["tickets"])
    return DrawSettlement(
        schema_version=payload["schema_version"],
        lottery=payload["lottery"],
        draw_number=int(payload["draw_number"]),
        draw_date=payload["draw_date"],
        prediction_record_path=payload["prediction_record_path"],
        prediction_generated_at=payload["prediction_generated_at"],
        prediction_dataset_hash=payload["prediction_dataset_hash"],
        settled_at=payload["settled_at"],
        actual_main_numbers=tuple(payload["actual_main_numbers"]),
        actual_bonus_numbers=tuple(payload["actual_bonus_numbers"]),
        payouts=payouts,
        tickets=tickets,
        ticket_count=int(payload["ticket_count"]),
        ticket_price_yen=int(payload["ticket_price_yen"]),
        paper_total_cost_yen=int(payload["paper_total_cost_yen"]),
        paper_gross_winnings_yen=payload["paper_gross_winnings_yen"],
        paper_net_yen=payload["paper_net_yen"],
        financial_status=payload["financial_status"],
        warnings=tuple(payload["warnings"]),
    )


def add_manual_payout(
    lottery: LotteryDefinition,
    *,
    draw_number: int,
    prize_tier: str,
    payout_yen: int,
    winners_count: int | None = None,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    confirmed: bool = False,
) -> DrawSettlement:
    if not confirmed:
        raise ResearchValidationError("manual payout entry requires --confirm-manual")
    path = settlement_path(settlement_root, lottery, draw_number)
    if not path.exists():
        raise ResearchValidationError("manual payout requires an existing evaluated settlement")
    existing = load_settlement(path)
    incoming = manual_draw_payout(
        lottery,
        draw_number=draw_number,
        prize_tier=prize_tier,
        payout_yen=payout_yen,
        winners_count=winners_count,
    )
    payouts = merge_draw_payouts(existing.payouts, (incoming,), lottery)
    completed = _with_payouts(existing, lottery, payouts)
    save_settlement(completed, path)
    write_financial_ledger(settlement_root)
    return completed


def financial_summary(
    *,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    lottery: LotteryDefinition | None = None,
    on_date: date | None = None,
    month: str | None = None,
) -> dict[str, Any]:
    settlements = tuple(_iter_settlements(settlement_root, lottery))
    filtered = tuple(
        settlement
        for settlement in settlements
        if _matches_period(settlement, on_date=on_date, month=month)
    )
    return _summary_payload(filtered, lottery_code=str(lottery.code) if lottery else "ALL")


def write_financial_ledger(settlement_root: str | Path = SETTLEMENT_ROOT) -> Path:
    root = Path(settlement_root)
    payload = {
        "schema_version": "stage12-paper-financial-ledger-v1",
        "all_time": financial_summary(settlement_root=root),
        "lotteries": {
            str(lottery.code): financial_summary(settlement_root=root, lottery=lottery)
            for lottery in (LOTO6, MINI_LOTO)
        },
    }
    path = root / "ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def settlement_from_prediction_and_draw(
    record: PredictionRecord,
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
    prediction_record_path: str,
    payouts: tuple[DrawPayout, ...],
) -> DrawSettlement:
    # Utility for tests and future callers when only the draw object is available.
    if record.evaluation is None:
        raise ResearchValidationError("prediction record has no evaluation")
    if draw.draw_number != record.target_draw_number:
        raise ResearchValidationError("draw does not match prediction target")
    return build_settlement(record, lottery, prediction_record_path, payouts)


def _with_payouts(
    settlement: DrawSettlement,
    lottery: LotteryDefinition,
    payouts: tuple[DrawPayout, ...],
) -> DrawSettlement:
    payout_by_tier = {payout.prize_tier: payout for payout in payouts}
    tickets: list[TicketSettlement] = []
    payout_pending = False
    for ticket in settlement.tickets:
        payout_yen = ticket.payout_yen
        if ticket.prize_tier != NO_PRIZE:
            payout = payout_by_tier.get(ticket.prize_tier)
            if payout is None:
                payout_yen = None
                payout_pending = True
            else:
                payout_yen = payout.payout_yen
        tickets.append(
            TicketSettlement(
                ticket.ticket_index,
                ticket.numbers,
                ticket.main_match_count,
                ticket.bonus_match_count,
                ticket.prize_tier,
                payout_yen,
            )
        )
    gross = None if payout_pending else sum(ticket.payout_yen or 0 for ticket in tickets)
    net = None if gross is None else gross - settlement.paper_total_cost_yen
    return replace(
        settlement,
        settled_at=datetime.now(UTC).isoformat(),
        payouts=payouts,
        tickets=tuple(tickets),
        paper_gross_winnings_yen=gross,
        paper_net_yen=net,
        financial_status=FINANCIAL_STATUS_PAYOUT_PENDING
        if payout_pending
        else FINANCIAL_STATUS_COMPLETE,
    )


def _try_collect_payouts(lottery: LotteryDefinition, draw_number: int) -> tuple[DrawPayout, ...]:
    try:
        return collect_smbc_draw_payouts(lottery, draw_number)
    except (KeyError, ResearchValidationError, ValueError):
        return ()


def _iter_settlements(
    settlement_root: str | Path,
    lottery: LotteryDefinition | None,
) -> tuple[DrawSettlement, ...]:
    root = Path(settlement_root)
    directories = (
        (settlement_lottery_dir(root, lottery),)
        if lottery is not None
        else tuple(settlement_lottery_dir(root, known) for known in (LOTO6, MINI_LOTO))
    )
    settlements: list[DrawSettlement] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            settlements.append(load_settlement(path))
    return tuple(settlements)


def _summary_payload(
    settlements: tuple[DrawSettlement, ...],
    *,
    lottery_code: str,
) -> dict[str, Any]:
    complete = tuple(
        settlement
        for settlement in settlements
        if settlement.financial_status == FINANCIAL_STATUS_COMPLETE
    )
    tickets = sum(settlement.ticket_count for settlement in complete)
    cost = sum(settlement.paper_total_cost_yen for settlement in complete)
    gross = sum(settlement.paper_gross_winnings_yen or 0 for settlement in complete)
    prize_counts: dict[str, int] = {}
    winning_ticket_count = 0
    for settlement in complete:
        for ticket in settlement.tickets:
            prize_counts[ticket.prize_tier] = prize_counts.get(ticket.prize_tier, 0) + 1
            if ticket.prize_tier != NO_PRIZE:
                winning_ticket_count += 1
    return {
        "lottery": lottery_code,
        "draws_evaluated": len(complete),
        "payout_pending_draws": len(settlements) - len(complete),
        "tickets": tickets,
        "paper_total_cost_yen": cost,
        "paper_gross_winnings_yen": gross,
        "paper_net_yen": gross - cost,
        "winning_ticket_count": winning_ticket_count,
        "prize_tier_counts": dict(sorted(prize_counts.items())),
        "paper_return_ratio": None if cost == 0 else gross / cost,
    }


def _matches_period(
    settlement: DrawSettlement,
    *,
    on_date: date | None,
    month: str | None,
) -> bool:
    draw_date = date.fromisoformat(settlement.draw_date)
    if on_date is not None and draw_date != on_date:
        return False
    if month is not None and settlement.draw_date[:7] != month:
        return False
    return True


def _settlement_scientific_payload(settlement: DrawSettlement) -> dict[str, Any]:
    payload = to_jsonable(settlement)
    payload.pop("settled_at", None)
    return payload
