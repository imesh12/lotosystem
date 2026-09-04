from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.automation import (
    ACTION_CHECK_RESULT,
    ACTION_NO_ACTION,
    ACTION_PREDICTION_CREATED,
    ACTION_RESULT_PROCESSED,
    ACTION_SOURCE_FAILURE,
    automation_status,
    run_automation_once,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    HISTORY_UPDATE_NEW_RESULT,
    HISTORY_UPDATE_NO_NEW_RESULT,
    write_canonical_history_csv,
)
from backend.app.research.operational_cycle import (
    CycleHistorySummary,
    CycleNextPredictionSummary,
    OperationalCycleResult,
)
from backend.app.research.production import evaluate_pending_predictions, generate_next_prediction
from backend.app.research.settlement import (
    FINANCIAL_STATUS_PAYOUT_PENDING,
    load_settlement,
    settlement_path,
)


def _draws(
    lottery: LotteryDefinition,
    *,
    start_number: int,
    count: int,
    start_date: date,
) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    step = 3 if str(lottery.code) == "LOTO6" else 7
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
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
                lottery=lottery,
                draw_number=start_number + index,
                draw_date=start_date + timedelta(days=index * step),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _patch_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lottery: LotteryDefinition,
    draws: tuple[HistoricalDraw, ...],
) -> Path:
    path = tmp_path / f"{lottery.code}.csv"
    write_canonical_history_csv(draws, path)
    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", lambda _: path)
    return path


def _cycle_result(
    lottery: LotteryDefinition,
    *,
    status: str,
    appended: int,
    evaluated: tuple[int, ...] = (),
    settlements: tuple[str, ...] = (),
) -> OperationalCycleResult:
    return OperationalCycleResult(
        lottery=str(lottery.code),
        cycle_id=f"CYCLE-{lottery.code}-TEST",
        history=CycleHistorySummary(
            previous_latest_draw=100,
            new_latest_draw=100 + appended,
            appended=appended,
            output_path="history.csv",
            update_status=status,
            selected_source="secondary",
            fallback_used=True,
            source_attempts=(),
        ),
        evaluated_predictions=evaluated,
        settlements=settlements,
        next_prediction=CycleNextPredictionSummary(
            draw=101 + appended,
            target_date="2026-08-27",
            status="PENDING",
            tickets=3,
            record_path="prediction.json",
            created=appended > 0,
        ),
        stage27=None,
        cycle_record_path="cycle.json",
        errors=(),
        warnings=(),
    )


@pytest.mark.parametrize(
    ("latest_date", "target_date"),
    [(date(2026, 8, 20), "2026-08-24"), (date(2026, 8, 24), "2026-08-27")],
)
def test_loto6_draw_days_are_due_after_check_window(
    latest_date: date,
    target_date: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(
        LOTO6,
        start_number=2000,
        count=110,
        start_date=latest_date - timedelta(days=327),
    )
    draws = (*draws[:-1], _replace_draw_date(draws[-1], latest_date))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    generate_next_prediction(draws, LOTO6, ResearchConfig(seed=123456), prediction_root=tmp_path)

    payload = automation_status(
        lottery=LOTO6,
        prediction_root=tmp_path,
        now=datetime.fromisoformat(f"{target_date}T21:30:00+09:00"),
    )

    assert payload["lotteries"]["LOTO6"]["next_action"] == ACTION_CHECK_RESULT


def test_mini_loto_tuesday_due_and_non_draw_future_waits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(MINI_LOTO, start_number=900, count=110, start_date=date(2024, 7, 23))
    draws = (*draws[:-1], _replace_draw_date(draws[-1], date(2026, 8, 18)))
    _patch_history(monkeypatch, tmp_path, MINI_LOTO, draws)
    generate_next_prediction(
        draws,
        MINI_LOTO,
        ResearchConfig(seed=123456),
        prediction_root=tmp_path,
    )

    due = automation_status(
        lottery=MINI_LOTO,
        prediction_root=tmp_path,
        now=datetime(2026, 8, 25, 21, 15, tzinfo=UTC),
    )
    waiting = automation_status(
        lottery=MINI_LOTO,
        prediction_root=tmp_path,
        now=datetime.fromisoformat("2026-08-24T21:15:00+09:00"),
    )

    assert due["lotteries"]["MINI_LOTO"]["next_action"] == ACTION_CHECK_RESULT
    assert waiting["lotteries"]["MINI_LOTO"]["next_action"] == ACTION_NO_ACTION


def test_before_result_check_window_is_no_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    generate_next_prediction(draws, LOTO6, ResearchConfig(seed=123456), prediction_root=tmp_path)

    payload = automation_status(
        lottery=LOTO6,
        prediction_root=tmp_path,
        now=datetime.fromisoformat("2026-08-24T20:59:00+09:00"),
    )

    assert payload["lotteries"]["LOTO6"]["next_action"] == ACTION_NO_ACTION


def test_auto_run_processes_new_result_and_payout_pending_does_not_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    generate_next_prediction(draws, LOTO6, ResearchConfig(seed=123456), prediction_root=tmp_path)

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        return _cycle_result(
            LOTO6,
            status=HISTORY_UPDATE_NEW_RESULT,
            appended=1,
            evaluated=(2110,),
            settlements=("settlement.json",),
        )

    payload = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-08-24T21:30:00+09:00"),
        cycle_runner=runner,
    )

    loto = payload["lotteries"][0]
    assert loto["action"] == ACTION_RESULT_PROCESSED
    assert loto["prediction_evaluation"]["evaluated"] == (2110,)
    assert loto["settlement"]["paths"] == ("settlement.json",)


def test_auto_run_no_new_result_and_second_run_are_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    generate_next_prediction(draws, LOTO6, ResearchConfig(seed=123456), prediction_root=tmp_path)

    calls = 0

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        nonlocal calls
        calls += 1
        return _cycle_result(LOTO6, status=HISTORY_UPDATE_NO_NEW_RESULT, appended=0)

    first = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-08-24T21:30:00+09:00"),
        cycle_runner=runner,
    )
    second = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-08-24T21:31:00+09:00"),
        cycle_runner=runner,
    )

    assert calls == 2
    assert first["lotteries"][0]["action"] == ACTION_CHECK_RESULT
    assert second["lotteries"][0]["history_update"]["appended"] == 0


def test_auto_run_no_action_reconciles_evaluated_prediction_missing_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=112, start_date=date(2025, 9, 25))
    history = _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    prediction_root = tmp_path / "predictions"
    settlement_root = tmp_path / "settlements"
    generated = generate_next_prediction(
        draws[:-1],
        LOTO6,
        ResearchConfig(seed=123456),
        tickets_per_draw=3,
        prediction_root=prediction_root,
    )
    evaluate_pending_predictions(draws, LOTO6, prediction_root=prediction_root)
    generate_next_prediction(
        draws,
        LOTO6,
        ResearchConfig(seed=123456),
        tickets_per_draw=3,
        prediction_root=prediction_root,
    )
    monkeypatch.setattr("backend.app.research.settlement.collect_smbc_draw_payouts", lambda *_: ())

    first = run_automation_once(
        lottery=LOTO6,
        prediction_root=prediction_root,
        settlement_root=settlement_root,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat(f"{generated.record.target_draw_date}T12:00:00+09:00"),
    )
    run_automation_once(
        lottery=LOTO6,
        prediction_root=prediction_root,
        settlement_root=settlement_root,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat(f"{generated.record.target_draw_date}T12:01:00+09:00"),
    )

    expected_path = settlement_path(settlement_root, LOTO6, generated.record.target_draw_number)
    settlement = load_settlement(expected_path)
    assert history.exists()
    assert first["lotteries"][0]["action"] == ACTION_NO_ACTION
    assert first["lotteries"][0]["settlement"]["paths"] == (str(expected_path),)
    assert tuple((settlement_root / "LOTO6").glob("*.json")) == (expected_path,)
    assert settlement.paper_total_cost_yen == 600


def test_source_failure_returns_retry_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    generate_next_prediction(draws, LOTO6, ResearchConfig(seed=123456), prediction_root=tmp_path)

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        raise ResearchValidationError("source unavailable")

    payload = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-08-24T21:30:00+09:00"),
        cycle_runner=runner,
    )

    assert payload["lotteries"][0]["action"] == ACTION_SOURCE_FAILURE
    assert payload["lotteries"][0]["errors"] == ("source unavailable",)


def test_concurrent_and_stale_locks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29))
    _patch_history(monkeypatch, tmp_path, LOTO6, draws)
    root = tmp_path / "automation"
    root.mkdir()
    lock = root / "automation.lock"
    lock.write_text("active", encoding="utf-8")

    with pytest.raises(ResearchValidationError, match="automation lock is active"):
        run_automation_once(
            lottery=LOTO6,
            automation_root=root,
            notification_root=tmp_path / "notifications",
        )

    old = datetime(2026, 8, 23, 0, 0, tzinfo=UTC).timestamp()
    os.utime(lock, (old, old))
    payload = run_automation_once(
        lottery=LOTO6,
        prediction_root=tmp_path,
        automation_root=root,
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-08-24T10:00:00+09:00"),
    )
    assert payload["lotteries"][0]["action"] == ACTION_PREDICTION_CREATED
    assert not lock.exists()


def test_all_lotteries_status_and_financial_pending_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loto_history = tmp_path / "loto6.csv"
    mini_history = tmp_path / "mini.csv"
    write_canonical_history_csv(
        _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 9, 29)),
        loto_history,
    )
    write_canonical_history_csv(
        _draws(MINI_LOTO, start_number=900, count=110, start_date=date(2024, 7, 23)),
        mini_history,
    )

    def history_path(lottery: LotteryDefinition) -> Path:
        return loto_history if lottery.code == LOTO6.code else mini_history

    monkeypatch.setattr("backend.app.research.automation.canonical_history_path", history_path)
    settlement_dir = tmp_path / "settlements" / "LOTO6"
    settlement_dir.mkdir(parents=True)
    (settlement_dir / "2131.json").write_text(
        '{"financial_status": "' + FINANCIAL_STATUS_PAYOUT_PENDING + '"}',
        encoding="utf-8",
    )

    payload = automation_status(
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        now=datetime.fromisoformat("2026-08-24T10:00:00+09:00"),
    )

    assert set(payload["lotteries"]) == {"LOTO6", "MINI_LOTO"}
    assert payload["financial_pending_count"] == 1


def _replace_draw_date(draw: HistoricalDraw, draw_date: date) -> HistoricalDraw:
    return HistoricalDraw(
        draw.lottery,
        draw.draw_number,
        draw_date,
        draw.main_numbers,
        draw.bonus_numbers,
        source=draw.source,
        source_url=draw.source_url,
        retrieved_at=draw.retrieved_at,
        content_hash=draw.content_hash,
    )
