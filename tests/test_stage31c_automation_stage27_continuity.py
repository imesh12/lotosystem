from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.automation import (
    ACTION_RESULT_PROCESSED,
    ACTION_SOURCE_FAILURE,
    STAGE27_LIFECYCLE_ERROR,
    STAGE27_LIFECYCLE_OK,
    run_automation_once,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    HISTORY_UPDATE_NEW_RESULT,
    HISTORY_UPDATE_NO_NEW_RESULT,
    HistoryUpdateResult,
    merge_historical_draws,
    verify_history,
    write_canonical_history_csv,
)
from backend.app.research.operational_cycle import (
    CycleHistorySummary,
    CycleNextPredictionSummary,
    OperationalCycleResult,
    run_post_draw_cycle,
)
from backend.app.research.production import generate_next_prediction, load_prediction_record
from backend.app.research.stage27_prospective_signals import (
    load_stage27_record,
    stage27_record_path,
)

# ---------------------------------------------------------------------------
# Fixtures shared with the Stage 11 / Stage 13 test suites (same patterns,
# reused rather than reinvented). All dates are deliberately far in the past
# relative to the "now" timestamps used below, so every automation call is
# unambiguously "due" without depending on next_scheduled_draw_date's exact
# weekly-cadence arithmetic.
# ---------------------------------------------------------------------------

_START_DATE = date(2024, 1, 2)


def _draws(
    lottery: LotteryDefinition,
    *,
    start_number: int,
    count: int,
    start_date: date = _START_DATE,
) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    step = 7
    stride = 4
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
    history_path: Path,
) -> None:
    monkeypatch.setattr(
        "backend.app.research.automation.canonical_history_path",
        lambda _: history_path,
    )


@dataclass(frozen=True, slots=True)
class _FakeUpdater:
    """Mirrors tests/test_stage11_operational_cycle.py's _FakeUpdater: appends
    a fixed set of draws instead of making any network call."""

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
            update_status=HISTORY_UPDATE_NEW_RESULT if appended else HISTORY_UPDATE_NO_NEW_RESULT,
        )


def _real_cycle_runner(
    history_path: Path,
    stage27_root: Path,
    draws_to_fetch: tuple[HistoricalDraw, ...],
):
    """Builds a `cycle_runner` for run_automation_once that delegates to the
    REAL run_post_draw_cycle (hence the REAL Stage 27 lifecycle), with the
    network-touching history fetch replaced by a deterministic fake and an
    isolated Stage 27 root -- never the live repository's data directories.
    """

    def runner(
        lottery: LotteryDefinition,
        config: ResearchConfig,
        *,
        tickets_per_draw: int,
        prediction_root: Path,
        settlement_root: Path,
        headed: bool,
        row_timeout_ms: int,
        result_source_order: tuple[str, ...] | None,
        started_at: datetime,
    ) -> OperationalCycleResult:
        return run_post_draw_cycle(
            lottery,
            config,
            history_path=history_path,
            prediction_root=prediction_root,
            settlement_root=settlement_root,
            tickets_per_draw=tickets_per_draw,
            started_at=started_at,
            history_updater=_FakeUpdater(history_path, draws_to_fetch),
            stage27_root=stage27_root,
        )

    return runner


def _cycle_result_with_stage27(
    *,
    status: str,
    appended: int,
    stage27: dict[str, object] | None,
) -> OperationalCycleResult:
    return OperationalCycleResult(
        lottery=str(MINI_LOTO.code),
        cycle_id="CYCLE-MINI_LOTO-TEST",
        history=CycleHistorySummary(
            previous_latest_draw=1406,
            new_latest_draw=1406 + appended,
            appended=appended,
            output_path="history.csv",
            update_status=status,
            selected_source="secondary",
            fallback_used=True,
            source_attempts=(),
        ),
        evaluated_predictions=(),
        settlements=(),
        next_prediction=CycleNextPredictionSummary(
            draw=1407 + appended,
            target_date="2026-10-06",
            status="PENDING",
            tickets=3,
            record_path="prediction.json",
            created=appended > 0,
        ),
        stage27=stage27,
        cycle_record_path="cycle.json",
        errors=(),
        warnings=(),
    )


# ---------------------------------------------------------------------------
# D / E: Stage 27 lifecycle must be surfaced, never silently dropped.
# ---------------------------------------------------------------------------


def test_stage27_lifecycle_surfaced_and_freeze_happens_automatically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real post-draw cycle run through automation must both freeze the
    next Stage 27 target automatically and surface that outcome in the
    automation payload -- this is the Section D/E production-code fix."""
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=110)
    write_canonical_history_csv(draws[:-1], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-1], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )
    stage27_root = tmp_path / "stage27"

    runner = _real_cycle_runner(history_path, stage27_root, (draws[-1],))

    payload = run_automation_once(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=runner,
    )

    mini = payload["lotteries"][0]
    assert mini["action"] == ACTION_RESULT_PROCESSED
    assert mini["stage27_lifecycle_status"] == STAGE27_LIFECYCLE_OK
    assert mini["stage27"] is not None
    assert mini["errors"] == ()

    next_target = draws[-1].draw_number + 1
    frozen_path = stage27_record_path(stage27_root, MINI_LOTO, next_target)
    assert frozen_path.exists(), "Stage 27 must freeze the next target automatically"
    frozen = load_stage27_record(frozen_path)
    assert frozen.history_cutoff_draw == draws[-1].draw_number
    assert frozen.draw_number == next_target

    saved_run = json.loads(Path(payload["record_path"]).read_text(encoding="utf-8"))
    assert saved_run["lotteries"][0]["stage27_lifecycle_status"] == STAGE27_LIFECYCLE_OK


def test_stage27_lifecycle_failure_is_surfaced_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If Stage 27's sub-cycle fails, the automation run must not claim
    overall success while silently dropping that failure."""
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=110)
    write_canonical_history_csv(draws[:-1], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-1], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        return _cycle_result_with_stage27(
            status=HISTORY_UPDATE_NEW_RESULT,
            appended=1,
            stage27={
                "experiment": "stage27_prospective_signal_tracking",
                "status": "ERROR",
                "error": "synthetic freeze failure for test",
                "warnings": ("Stage 27 prospective tracking skipped: synthetic failure",),
            },
        )

    payload = run_automation_once(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=runner,
    )

    mini = payload["lotteries"][0]
    assert mini["stage27_lifecycle_status"] == STAGE27_LIFECYCLE_ERROR
    assert any("synthetic freeze failure" in error for error in mini["errors"])
    assert mini["stage27"]["status"] == "ERROR"
    # The run-level errors must also surface it (not just the per-lottery payload).
    assert any("synthetic freeze failure" in error for error in payload["errors"])


def test_source_failure_path_reports_stage27_not_applicable_for_mini_loto(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=110)
    write_canonical_history_csv(draws, history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws, MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )

    def runner(*args: object, **kwargs: object) -> OperationalCycleResult:
        raise ResearchValidationError("source unavailable")

    payload = run_automation_once(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=runner,
    )

    mini = payload["lotteries"][0]
    assert mini["action"] == ACTION_SOURCE_FAILURE
    assert mini["stage27"] is None
    assert mini["stage27_lifecycle_status"] == STAGE27_LIFECYCLE_ERROR


# ---------------------------------------------------------------------------
# F: idempotency and immutability through the automation entry point.
# ---------------------------------------------------------------------------


def test_duplicate_automation_execution_does_not_duplicate_stage27_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=110)
    write_canonical_history_csv(draws[:-1], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-1], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )
    stage27_root = tmp_path / "stage27"
    next_target = draws[-1].draw_number + 1

    common_kwargs = dict(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
    )

    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-1],)),
        **common_kwargs,
    )
    frozen_path = stage27_record_path(stage27_root, MINI_LOTO, next_target)
    before_bytes = frozen_path.read_bytes()
    before_record = load_stage27_record(frozen_path)
    siblings_before = sorted(p.name for p in frozen_path.parent.glob("*.json"))

    # Re-running with no new draw to fetch must not alter or duplicate the record.
    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:31:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, ()),
        **common_kwargs,
    )

    after_bytes = frozen_path.read_bytes()
    after_record = load_stage27_record(frozen_path)
    siblings_after = sorted(p.name for p in frozen_path.parent.glob("*.json"))
    assert before_bytes == after_bytes, "frozen ranking must be byte-identical after a rerun"
    assert before_record.freeze_hash == after_record.freeze_hash
    assert siblings_before == siblings_after, "no duplicate Stage 27 record was created"


def test_evaluated_record_immutable_after_later_cycle_and_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Freeze target A, then run a later cycle that evaluates A and freezes
    B; A's evaluated record must never change afterwards, including across
    a third, fully idempotent rerun."""
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=112)
    write_canonical_history_csv(draws[:-2], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-2], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )
    stage27_root = tmp_path / "stage27"
    target_a = draws[-2].draw_number + 1  # frozen by cycle 1, evaluated by cycle 2
    common_kwargs = dict(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
    )

    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-2],)),
        **common_kwargs,
    )
    run_automation_once(
        now=datetime.fromisoformat("2026-10-12T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-1],)),
        **common_kwargs,
    )

    record_a_path = stage27_record_path(stage27_root, MINI_LOTO, target_a)
    record_a = load_stage27_record(record_a_path)
    assert record_a.status == "EVALUATED"
    evaluated_bytes = record_a_path.read_bytes()

    # A third, idempotent rerun with no new draws must not touch A at all.
    run_automation_once(
        now=datetime.fromisoformat("2026-10-12T21:31:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, ()),
        **common_kwargs,
    )
    assert record_a_path.read_bytes() == evaluated_bytes


def test_production_prediction_unchanged_across_idempotent_automation_reruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=110)
    write_canonical_history_csv(draws[:-1], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-1], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )
    stage27_root = tmp_path / "stage27"
    next_target = draws[-1].draw_number + 1
    common_kwargs = dict(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
    )

    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-1],)),
        **common_kwargs,
    )
    prediction_path = prediction_root / "MINI_LOTO" / f"{next_target}.json"
    before = load_prediction_record(prediction_path)

    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:31:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, ()),
        **common_kwargs,
    )
    after = load_prediction_record(prediction_path)

    assert before.tickets == after.tickets
    assert before.dataset_hash == after.dataset_hash
    assert before.generated_at == after.generated_at


def test_stage27_freeze_uses_only_history_through_latest_known_draw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No future-data leakage: the frozen target's cutoff must always equal
    the latest draw actually in canonical history at freeze time, across
    successive real automation cycles."""
    history_path = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=1400, count=112)
    write_canonical_history_csv(draws[:-2], history_path)
    _patch_history(monkeypatch, history_path)
    prediction_root = tmp_path / "predictions"
    generate_next_prediction(
        draws[:-2], MINI_LOTO, ResearchConfig(seed=123456), prediction_root=prediction_root
    )
    stage27_root = tmp_path / "stage27"
    common_kwargs = dict(
        lottery=MINI_LOTO,
        prediction_root=prediction_root,
        settlement_root=tmp_path / "settlements",
        automation_root=tmp_path / "automation",
        notification_root=tmp_path / "notifications",
    )

    run_automation_once(
        now=datetime.fromisoformat("2026-10-05T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-2],)),
        **common_kwargs,
    )
    first_target = draws[-2].draw_number + 1
    first_record = load_stage27_record(stage27_record_path(stage27_root, MINI_LOTO, first_target))
    assert first_record.history_cutoff_draw == draws[-2].draw_number

    run_automation_once(
        now=datetime.fromisoformat("2026-10-12T21:30:00+09:00"),
        cycle_runner=_real_cycle_runner(history_path, stage27_root, (draws[-1],)),
        **common_kwargs,
    )
    second_target = draws[-1].draw_number + 1
    second_record = load_stage27_record(stage27_record_path(stage27_root, MINI_LOTO, second_target))
    assert second_record.history_cutoff_draw == draws[-1].draw_number
    assert second_record.history_cutoff_draw < second_record.draw_number


# ---------------------------------------------------------------------------
# Backward compatibility with pre-fix AUTO-*.json records.
# ---------------------------------------------------------------------------


def test_old_automation_record_schema_without_stage27_keys_still_parses(tmp_path: Path) -> None:
    """A run record saved before this fix has no "stage27" /
    "stage27_lifecycle_status" keys at all. Anything reading saved records
    must tolerate that via .get(), never assume the keys exist."""
    old_style_record = {
        "run_id": "AUTO-OLD",
        "run_at": "2026-09-29T23:46:00+00:00",
        "completed_at": "2026-09-29T23:46:15+00:00",
        "timezone": "Asia/Tokyo",
        "lotteries": [
            {
                "lottery": "MINI_LOTO",
                "action": "RESULT_PROCESSED",
                "result_source_status": "NEW_RESULT",
                "history_update": {
                    "previous_latest_draw": 1405,
                    "new_latest_draw": 1406,
                    "appended": 1,
                },
                "prediction_evaluation": {"evaluated": (1406,)},
                "settlement": {"paths": ("settlement.json",)},
                "next_prediction": {"draw": 1407},
                "next_run_at": "2026-10-06T21:00:00+09:00",
                "warnings": (),
                "errors": (),
            }
        ],
        "warnings": (),
        "errors": (),
    }
    path = tmp_path / "AUTO-OLD.json"
    path.write_text(json.dumps(old_style_record, default=str), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    mini = reloaded["lotteries"][0]
    assert "stage27" not in mini
    # A consumer written against the new schema must use .get() and degrade
    # gracefully on an old record rather than raising KeyError.
    assert mini.get("stage27") is None
    assert mini.get("stage27_lifecycle_status") is None
