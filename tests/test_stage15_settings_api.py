from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.automation import ACTION_NO_ACTION, run_automation_once
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import write_canonical_history_csv
from backend.app.research.payouts import manual_draw_payout
from backend.app.research.prize import match_ticket
from backend.app.research.production import (
    PredictionRecord,
    PredictionTicket,
    evaluate_pending_predictions,
    generate_next_prediction,
    save_prediction_record,
)
from backend.app.research.settings import (
    load_settings,
    lottery_settings,
    update_settings,
)
from backend.app.research.settlement import build_settlement, save_settlement, settlement_path


def _draws(
    lottery: LotteryDefinition,
    *,
    start_number: int,
    count: int,
    start_date: date,
) -> tuple[HistoricalDraw, ...]:
    step = 3 if lottery.code == LOTO6.code else 7
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % (lottery.number_max - lottery.numbers_per_ticket)) + 1
        main = tuple(
            sorted(
                ((start + offset * stride - 1) % lottery.number_max) + 1
                for offset in range(lottery.numbers_per_ticket)
            )
        )
        bonus = next(
            number
            for number in range(lottery.number_min, lottery.number_max + 1)
            if number not in main
        )
        draws.append(
            HistoricalDraw(
                lottery,
                start_number + index,
                start_date + timedelta(days=index * step),
                main,
                (bonus,),
            )
        )
    return tuple(draws)


def _record_for_draw(
    lottery: LotteryDefinition,
    draw: HistoricalDraw,
    ticket: tuple[int, ...],
) -> PredictionRecord:
    ticket_record = PredictionTicket(1, lottery.validate_main_numbers(ticket), 0.0)
    match = match_ticket(ticket_record.numbers, draw, lottery)
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
        tickets_per_draw=1,
        ticket_price_yen=lottery.ticket_price_yen,
        cost_yen=lottery.ticket_price_yen,
        gross_winnings_yen=None,
        sklearn_version="test",
        config={},
        tickets=(ticket_record,),
        evaluation={
            "evaluated_at": "2026-08-24T01:00:00+00:00",
            "actual_draw_number": draw.draw_number,
            "actual_draw_date": draw.draw_date.isoformat(),
            "actual_main_numbers": draw.main_numbers,
            "actual_bonus_numbers": draw.bonus_numbers,
            "ticket_results": (
                {
                    "ticket_index": 1,
                    "numbers": ticket_record.numbers,
                    "main_match_count": match.main_match_count,
                    "bonus_match_count": match.bonus_match_count,
                    "prize_category": match.prize_name,
                    "qualifies_for_prize": match.qualifies_for_prize,
                },
            ),
            "best_match_count": match.main_match_count,
            "prize_qualified_ticket_count": int(match.qualifies_for_prize),
            "gross_winnings_yen": None,
        },
        warnings=(),
    )


def test_default_settings_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    defaults = load_settings(path)
    updated = update_settings(
        {"LOTO6": {"tickets_per_draw": 5}, "email_enabled": True},
        path=path,
    )
    reloaded = load_settings(path)

    assert defaults.loto6.tickets_per_draw == 3
    assert defaults.mini_loto.tickets_per_draw == 3
    assert updated.email_enabled is True
    assert reloaded.loto6.tickets_per_draw == 5


@pytest.mark.parametrize("count", [0, -1, 21])
def test_invalid_ticket_count(count: int, tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError):
        update_settings({"LOTO6": {"tickets_per_draw": count}}, path=tmp_path / "settings.json")


def test_unknown_settings_field_rejected(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError, match="unknown settings"):
        update_settings(
            {"LOTO6": {"tickets_per_draw": 3, "secret": "nope"}},
            path=tmp_path / "settings.json",
        )


def test_existing_prediction_immutable_and_future_prediction_uses_new_ticket_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=112, start_date=date(2025, 9, 25))
    history = tmp_path / "loto6.csv"
    write_canonical_history_csv(draws[:-1], history)
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", lambda _: history)
    prediction_root = tmp_path / "predictions"
    generated = generate_next_prediction(
        draws[:-1],
        LOTO6,
        ResearchConfig(seed=123456),
        tickets_per_draw=3,
        prediction_root=prediction_root,
    )
    settings_path = tmp_path / "settings.json"
    update_settings({"LOTO6": {"tickets_per_draw": 5}}, path=settings_path)

    existing = run_automation_once(
        lottery=LOTO6,
        prediction_root=prediction_root,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        settings_path=settings_path,
        now=datetime.fromisoformat(f"{generated.record.target_draw_date}T12:00:00+09:00"),
    )
    assert existing["lotteries"][0]["next_prediction"]["tickets"] == 3

    write_canonical_history_csv(draws, history)
    evaluate_pending_predictions(draws, LOTO6, prediction_root=prediction_root)
    future = run_automation_once(
        lottery=LOTO6,
        prediction_root=prediction_root,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        settings_path=settings_path,
        now=datetime.fromisoformat(f"{generated.record.target_draw_date}T12:01:00+09:00"),
    )

    assert future["lotteries"][0]["next_prediction"]["tickets"] == 5


def test_lottery_disabled_skips_automation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(MINI_LOTO, start_number=900, count=110, start_date=date(2024, 7, 23))
    history = tmp_path / "mini.csv"
    write_canonical_history_csv(draws, history)
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", lambda _: history)
    settings_path = tmp_path / "settings.json"
    update_settings({"MINI_LOTO": {"enabled": False}}, path=settings_path)

    result = run_automation_once(
        lottery=MINI_LOTO,
        prediction_root=tmp_path / "predictions",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        settings_path=settings_path,
        now=datetime.fromisoformat(f"{draws[-1].draw_date.isoformat()}T12:00:00+09:00"),
    )

    assert lottery_settings(MINI_LOTO, path=settings_path).enabled is False
    assert result["lotteries"][0]["action"] == ACTION_NO_ACTION
    assert "disabled" in result["lotteries"][0]["warnings"][0]


def test_api_settings_get_put_and_secret_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr("backend.app.api.routes.operations.API_SETTINGS_PATH", settings_path)

    get_response = client.get("/api/settings")
    put_response = client.put(
        "/api/settings",
        json={"LOTO6": {"tickets_per_draw": 5}, "email_enabled": True},
    )

    assert get_response.status_code == 200
    assert put_response.status_code == 200
    assert put_response.json()["LOTO6"]["tickets_per_draw"] == 5
    assert "SMTP" not in str(put_response.json()).upper()


def test_api_rejects_invalid_settings(client: TestClient) -> None:
    response = client.put("/api/settings", json={"LOTO6": {"tickets_per_draw": 0}})

    assert response.status_code == 400


def test_api_lottery_latest_next_history_financial_and_notifications(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2130, count=2, start_date=date(2026, 8, 20))
    history = tmp_path / "loto6.csv"
    write_canonical_history_csv(draws, history)
    monkeypatch.setattr("backend.app.research.operations.canonical_history_path", lambda _: history)
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", lambda _: history)
    monkeypatch.setattr("backend.app.research.operations.OPERATION_PREDICTION_ROOT", tmp_path / "p")
    monkeypatch.setattr("backend.app.research.operations.OPERATION_SETTLEMENT_ROOT", tmp_path / "s")
    monkeypatch.setattr(
        "backend.app.research.operations.notification_status",
        lambda: {
            "enabled": False,
            "pending": 0,
            "failed": 0,
            "sent": 0,
            "latest_notification": None,
        },
    )
    record = _record_for_draw(LOTO6, draws[-1], draws[-1].main_numbers)
    prediction_path = tmp_path / "p" / "LOTO6" / f"{draws[-1].draw_number}.json"
    save_prediction_record(record, prediction_path)
    settlement = build_settlement(
        record,
        LOTO6,
        str(prediction_path),
        (
            manual_draw_payout(
                LOTO6, draw_number=draws[-1].draw_number, prize_tier="1st", payout_yen=1
            ),
        ),
    )
    save_settlement(settlement, settlement_path(tmp_path / "s", LOTO6, draws[-1].draw_number))
    pending = replace(
        record,
        status="PENDING",
        target_draw_number=draws[-1].draw_number + 1,
        target_draw_date="2026-08-27",
        evaluation=None,
    )
    save_prediction_record(pending, tmp_path / "p" / "LOTO6" / "2132.json")

    latest = client.get("/api/lotteries/LOTO6/latest")
    next_prediction = client.get("/api/lotteries/LOTO6/next-prediction")
    history_response = client.get("/api/lotteries/LOTO6/history?limit=1&offset=0")
    financial = client.get("/api/financial/summary?lottery=LOTO6&period=all_time")
    notifications = client.get("/api/notifications/status")
    lotteries = client.get("/api/lotteries")

    assert latest.status_code == 200
    assert latest.json()["prediction_available"] is True
    assert next_prediction.json()["pending_prediction"]["ticket_count"] == 1
    assert history_response.json()["limit"] == 1
    assert history_response.json()["rows"][0]["prediction_available"] is True
    assert financial.json()["lottery"] == "LOTO6"
    assert notifications.json()["enabled"] is False
    assert lotteries.json()["lotteries"][0]["code"] == "LOTO6"


def test_api_status_and_missing_prediction(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    tmp_path: Path,
) -> None:
    draws = _draws(MINI_LOTO, start_number=1400, count=1, start_date=date(2026, 8, 18))
    mini_history = tmp_path / "mini.csv"
    loto_history = tmp_path / "loto6.csv"
    write_canonical_history_csv(draws, mini_history)
    write_canonical_history_csv(
        _draws(LOTO6, start_number=2130, count=1, start_date=date(2026, 8, 20)),
        loto_history,
    )

    def history_path(lottery: LotteryDefinition) -> Path:
        return mini_history if lottery.code == MINI_LOTO.code else loto_history

    monkeypatch.setattr("backend.app.research.operations.canonical_history_path", history_path)
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", history_path)
    monkeypatch.setattr("backend.app.research.operations.OPERATION_PREDICTION_ROOT", tmp_path / "p")
    monkeypatch.setattr("backend.app.research.operations.OPERATION_SETTLEMENT_ROOT", tmp_path / "s")

    status = client.get("/api/status")
    latest = client.get("/api/lotteries/MINI_LOTO/latest")

    assert status.status_code == 200
    assert "system" in status.json()
    assert "record_path" not in status.text
    assert "path" not in status.text.lower()
    assert latest.json()["prediction_available"] is False
