from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    HISTORY_UPDATE_NEW_RESULT,
    HistoryUpdateResult,
    merge_historical_draws,
    verify_history,
    write_canonical_history_csv,
)
from backend.app.research.operational_cycle import run_post_draw_cycle
from backend.app.research.payouts import DrawPayout, manual_draw_payout
from backend.app.research.prize import match_ticket
from backend.app.research.production import (
    PredictionRecord,
    PredictionTicket,
    generate_next_prediction,
    load_prediction_record,
    save_prediction_record,
)
from backend.app.research.settlement import (
    FINANCIAL_STATUS_COMPLETE,
    FINANCIAL_STATUS_PAYOUT_PENDING,
    NO_PRIZE,
    add_manual_payout,
    build_settlement,
    financial_summary,
    load_settlement,
    save_settlement,
    settlement_path,
)


def _record_for_tickets(
    lottery: LotteryDefinition,
    draw: HistoricalDraw,
    tickets: tuple[tuple[int, ...], ...],
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
        status="EVALUATED",
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
        evaluation={
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


def _payouts(
    lottery: LotteryDefinition,
    draw_number: int,
    values: dict[str, int],
) -> tuple[DrawPayout, ...]:
    return tuple(
        manual_draw_payout(
            lottery,
            draw_number=draw_number,
            prize_tier=tier,
            payout_yen=value,
            winners_count=1,
        )
        for tier, value in values.items()
    )


def test_loto6_prize_classifications_and_payouts() -> None:
    draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    record = _record_for_tickets(
        LOTO6,
        draw,
        (
            (1, 2, 3, 4, 5, 6),
            (1, 2, 3, 4, 5, 7),
            (1, 2, 3, 4, 5, 8),
            (1, 2, 3, 4, 8, 9),
            (1, 2, 3, 8, 9, 10),
            (8, 9, 10, 11, 12, 13),
        ),
    )
    settlement = build_settlement(
        record,
        LOTO6,
        "prediction.json",
        _payouts(
            LOTO6,
            2131,
            {"1st": 200_000_000, "2nd": 10_000_000, "3rd": 300_000, "4th": 6_800, "5th": 1_000},
        ),
    )

    assert [ticket.prize_tier for ticket in settlement.tickets] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        NO_PRIZE,
    ]
    assert settlement.paper_total_cost_yen == 1_200
    assert settlement.paper_gross_winnings_yen == 210_307_800
    assert settlement.paper_net_yen == 210_306_600
    assert settlement.financial_status == FINANCIAL_STATUS_COMPLETE


def test_mini_loto_prize_classifications_and_payouts() -> None:
    draw = HistoricalDraw(MINI_LOTO, 1401, date(2026, 8, 25), (1, 2, 3, 4, 5), (6,))
    record = _record_for_tickets(
        MINI_LOTO,
        draw,
        (
            (1, 2, 3, 4, 5),
            (1, 2, 3, 4, 6),
            (1, 2, 3, 4, 7),
            (1, 2, 3, 7, 8),
            (7, 8, 9, 10, 11),
        ),
    )
    settlement = build_settlement(
        record,
        MINI_LOTO,
        "prediction.json",
        _payouts(
            MINI_LOTO,
            1401,
            {"1st": 10_000_000, "2nd": 150_000, "3rd": 10_000, "4th": 1_000},
        ),
    )

    assert [ticket.prize_tier for ticket in settlement.tickets] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        NO_PRIZE,
    ]
    assert settlement.paper_gross_winnings_yen == 10_161_000
    assert settlement.paper_net_yen == 10_160_000


def test_zero_gross_and_negative_net_for_no_prize() -> None:
    draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    record = _record_for_tickets(LOTO6, draw, ((8, 9, 10, 11, 12, 13),))

    settlement = build_settlement(record, LOTO6, "prediction.json", ())

    assert settlement.financial_status == FINANCIAL_STATUS_COMPLETE
    assert settlement.paper_gross_winnings_yen == 0
    assert settlement.paper_net_yen == -200


def test_payout_pending_and_later_manual_completion(tmp_path: Path) -> None:
    draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    record = _record_for_tickets(LOTO6, draw, ((1, 2, 3, 8, 9, 10),))
    path = settlement_path(tmp_path, LOTO6, 2131)
    pending = build_settlement(record, LOTO6, "prediction.json", ())
    save_settlement(pending, path)

    completed = add_manual_payout(
        LOTO6,
        draw_number=2131,
        prize_tier="5",
        payout_yen=1_000,
        settlement_root=tmp_path,
        confirmed=True,
    )

    assert pending.financial_status == FINANCIAL_STATUS_PAYOUT_PENDING
    assert completed.financial_status == FINANCIAL_STATUS_COMPLETE
    assert completed.paper_gross_winnings_yen == 1_000
    assert completed.paper_net_yen == 800


def test_conflicting_payout_and_completed_settlement_rejected(tmp_path: Path) -> None:
    draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    record = _record_for_tickets(LOTO6, draw, ((1, 2, 3, 8, 9, 10),))
    settlement = build_settlement(
        record,
        LOTO6,
        "prediction.json",
        _payouts(LOTO6, 2131, {"5th": 1_000}),
    )
    path = settlement_path(tmp_path, LOTO6, 2131)
    save_settlement(settlement, path)
    save_settlement(settlement, path)

    conflicting = build_settlement(
        record,
        LOTO6,
        "prediction.json",
        _payouts(LOTO6, 2131, {"5th": 2_000}),
    )
    with pytest.raises(ResearchValidationError, match="conflicting completed settlement"):
        save_settlement(conflicting, path)
    with pytest.raises(ResearchValidationError, match="conflicting payout"):
        add_manual_payout(
            LOTO6,
            draw_number=2131,
            prize_tier="5th",
            payout_yen=2_000,
            settlement_root=tmp_path,
            confirmed=True,
        )


def test_financial_summary_all_time_date_month_and_combined(tmp_path: Path) -> None:
    loto_draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    mini_draw = HistoricalDraw(MINI_LOTO, 1401, date(2026, 8, 25), (1, 2, 3, 4, 5), (6,))
    loto = build_settlement(
        _record_for_tickets(LOTO6, loto_draw, ((1, 2, 3, 8, 9, 10),)),
        LOTO6,
        "loto.json",
        _payouts(LOTO6, 2131, {"5th": 1_000}),
    )
    mini = build_settlement(
        _record_for_tickets(MINI_LOTO, mini_draw, ((7, 8, 9, 10, 11),)),
        MINI_LOTO,
        "mini.json",
        (),
    )
    save_settlement(loto, settlement_path(tmp_path, LOTO6, 2131))
    save_settlement(mini, settlement_path(tmp_path, MINI_LOTO, 1401))

    loto_summary = financial_summary(settlement_root=tmp_path, lottery=LOTO6)
    combined = financial_summary(settlement_root=tmp_path)
    month = financial_summary(settlement_root=tmp_path, month="2026-08")
    today = financial_summary(settlement_root=tmp_path, on_date=date(2026, 8, 24))

    assert loto_summary["paper_gross_winnings_yen"] == 1_000
    assert combined["draws_evaluated"] == 2
    assert month["tickets"] == 2
    assert today["draws_evaluated"] == 1


def test_financial_summary_cli(tmp_path: Path) -> None:
    draw = HistoricalDraw(LOTO6, 2131, date(2026, 8, 24), (1, 2, 3, 4, 5, 6), (7,))
    settlement = build_settlement(
        _record_for_tickets(LOTO6, draw, ((1, 2, 3, 8, 9, 10),)),
        LOTO6,
        "prediction.json",
        _payouts(LOTO6, 2131, {"5th": 1_000}),
    )
    save_settlement(settlement, settlement_path(tmp_path, LOTO6, 2131))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--settlement-root",
            str(tmp_path),
            "financial-summary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["paper_gross_winnings_yen"] == 1_000
    assert payload["paper_return_ratio"] == 5


@dataclass(frozen=True, slots=True)
class _FakeUpdater:
    path: Path
    draws_to_fetch: tuple[HistoricalDraw, ...]

    def __call__(self, lottery: LotteryDefinition) -> HistoryUpdateResult:
        existing = load_draws_csv(self.path, lottery)
        merged, appended, unchanged = merge_historical_draws(existing, self.draws_to_fetch)
        write_canonical_history_csv(merged, self.path)
        return HistoryUpdateResult(
            output_path=str(self.path),
            fetched_count=len(self.draws_to_fetch),
            existing_count=len(existing),
            written_count=len(merged),
            appended_count=appended,
            unchanged_count=unchanged,
            verification=verify_history(merged, lottery),
            update_status=HISTORY_UPDATE_NEW_RESULT,
        )


def test_run_cycle_creates_payout_pending_settlement_and_next_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history = tmp_path / "loto6.csv"
    draws = []
    for index in range(112):
        draws.append(
            HistoricalDraw(
                LOTO6,
                2000 + index,
                date(2025, 1, 2) + timedelta(days=index * 3),
                (1, 2, 3, 4, 5, 6) if index == 110 else tuple(range(8, 14)),
                (7,),
            )
        )
    history_draws = tuple(draws)
    write_canonical_history_csv(history_draws[:-2], history)
    generated = generate_next_prediction(
        history_draws[:-2],
        LOTO6,
        ResearchConfig(seed=123456),
        tickets_per_draw=3,
        prediction_root=tmp_path / "predictions",
    )
    record = load_prediction_record(generated.record_path)
    record = replace(
        record,
        tickets=(PredictionTicket(1, (1, 2, 3, 8, 9, 10), 0.0), *record.tickets[1:]),
    )
    save_prediction_record(record, generated.record_path)
    monkeypatch.setattr("backend.app.research.settlement.collect_smbc_draw_payouts", lambda *_: ())

    result = run_post_draw_cycle(
        LOTO6,
        ResearchConfig(seed=123456),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=3,
        history_updater=_FakeUpdater(history, (history_draws[-2],)),
    )

    settlement = load_settlement(settlement_path(tmp_path / "settlements", LOTO6, 2110))
    assert result.next_prediction.created is True
    assert settlement.financial_status == FINANCIAL_STATUS_PAYOUT_PENDING
    assert settlement.paper_gross_winnings_yen is None
