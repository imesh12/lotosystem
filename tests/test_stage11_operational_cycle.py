from __future__ import annotations

from dataclasses import dataclass
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
    HISTORY_UPDATE_NO_NEW_RESULT,
    HISTORY_UPDATE_SOURCE_FAILURE,
    HistoryUpdateResult,
    merge_historical_draws,
    verify_history,
    write_canonical_history_csv,
)
from backend.app.research.operational_cycle import (
    read_cycle_record,
    run_post_draw_cycle,
)
from backend.app.research.production import (
    PREDICTION_STATUS_EVALUATED,
    PREDICTION_STATUS_PENDING,
    generate_next_prediction,
    load_prediction_record,
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
    modulo = lottery.number_max
    stride = 5 if lottery.numbers_per_ticket == 6 else 4
    for index in range(count):
        start = (index % (lottery.number_max - lottery.numbers_per_ticket)) + 1
        main = tuple(
            sorted(
                ((start + offset * stride - 1) % modulo) + 1
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


def _config() -> ResearchConfig:
    return ResearchConfig(seed=123456)


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
            update_status=HISTORY_UPDATE_NEW_RESULT if appended else HISTORY_UPDATE_NO_NEW_RESULT,
        )


def _write_history(path: Path, draws: tuple[HistoricalDraw, ...]) -> None:
    write_canonical_history_csv(draws, path)


def test_cycle_appends_evaluates_pending_and_generates_next(tmp_path: Path) -> None:
    history = tmp_path / "loto6.csv"
    draws = _draws(LOTO6, start_number=2000, count=112, start_date=date(2025, 1, 2))
    _write_history(history, draws[:-2])
    generate_next_prediction(
        draws[:-2],
        LOTO6,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path / "predictions",
    )

    result = run_post_draw_cycle(
        LOTO6,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=3,
        history_updater=_FakeUpdater(history, (draws[-2],)),
    )

    assert result.history.appended == 1
    assert result.history.update_status == HISTORY_UPDATE_NEW_RESULT
    assert result.evaluated_predictions == (draws[-2].draw_number,)
    assert result.next_prediction.draw == draws[-1].draw_number
    assert result.next_prediction.created is True
    evaluated = load_prediction_record(
        tmp_path / "predictions" / "LOTO6" / f"{draws[-2].draw_number}.json"
    )
    assert evaluated.status == PREDICTION_STATUS_EVALUATED
    assert evaluated.evaluation is not None
    assert "evaluated_at" in evaluated.evaluation
    assert evaluated.generated_at
    assert evaluated.cost_yen == 600


def test_cycle_no_new_result_is_idempotent_and_preserves_pending(tmp_path: Path) -> None:
    history = tmp_path / "loto6.csv"
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 1, 2))
    _write_history(history, draws)
    pending = generate_next_prediction(
        draws,
        LOTO6,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path / "predictions",
    )

    first = run_post_draw_cycle(
        LOTO6,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=3,
        history_updater=_FakeUpdater(history, ()),
    )
    second = run_post_draw_cycle(
        LOTO6,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=3,
        history_updater=_FakeUpdater(history, ()),
    )

    assert first.history.appended == 0
    assert second.history.appended == 0
    assert first.history.update_status == HISTORY_UPDATE_NO_NEW_RESULT
    assert second.history.update_status == HISTORY_UPDATE_NO_NEW_RESULT
    assert first.evaluated_predictions == ()
    assert second.evaluated_predictions == ()
    assert first.next_prediction.created is False
    assert second.next_prediction.created is False
    assert load_prediction_record(pending.record_path).status == PREDICTION_STATUS_PENDING


def test_cycle_catch_up_does_not_fabricate_retroactive_prediction(tmp_path: Path) -> None:
    history = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=900, count=113, start_date=date(2024, 1, 2))
    _write_history(history, draws[:-3])
    generate_next_prediction(
        draws[:-3],
        MINI_LOTO,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path / "predictions",
    )

    result = run_post_draw_cycle(
        MINI_LOTO,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=3,
        history_updater=_FakeUpdater(history, draws[-3:-1]),
    )

    assert result.history.appended == 2
    assert result.evaluated_predictions == (draws[-3].draw_number,)
    assert result.next_prediction.draw == draws[-1].draw_number
    missing_retroactive = tmp_path / "predictions" / "MINI_LOTO" / f"{draws[-2].draw_number}.json"
    assert not missing_retroactive.exists()


def test_cycle_preserves_existing_future_pending_with_changed_ticket_count(
    tmp_path: Path,
) -> None:
    history = tmp_path / "loto6.csv"
    draws = _draws(LOTO6, start_number=2000, count=110, start_date=date(2025, 1, 2))
    _write_history(history, draws)
    generate_next_prediction(
        draws,
        LOTO6,
        _config(),
        tickets_per_draw=5,
        prediction_root=tmp_path / "predictions",
    )

    result = run_post_draw_cycle(
        LOTO6,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=1,
        history_updater=_FakeUpdater(history, ()),
    )

    assert result.next_prediction.tickets == 5
    assert result.next_prediction.created is False


def test_cycle_failure_does_not_evaluate_or_generate_later_prediction(tmp_path: Path) -> None:
    history = tmp_path / "loto6.csv"
    draws = _draws(LOTO6, start_number=2000, count=111, start_date=date(2025, 1, 2))
    _write_history(history, draws[:-1])
    pending = generate_next_prediction(
        draws[:-1],
        LOTO6,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path / "predictions",
    )

    def failing_updater(lottery: LotteryDefinition) -> HistoryUpdateResult:
        raise ResearchValidationError("browser update failed")

    with pytest.raises(ResearchValidationError, match="browser update failed"):
        run_post_draw_cycle(
            LOTO6,
            _config(),
            history_path=history,
            prediction_root=tmp_path / "predictions",
            settlement_root=tmp_path / "settlements",
            tickets_per_draw=3,
            history_updater=failing_updater,
        )

    assert load_prediction_record(pending.record_path).status == PREDICTION_STATUS_PENDING
    assert not (tmp_path / "predictions" / "LOTO6" / f"{draws[-1].draw_number + 1}.json").exists()
    cycle_records = list((tmp_path / "predictions" / "LOTO6" / "cycles").glob("*.json"))
    assert cycle_records
    cycle_record = read_cycle_record(cycle_records[0])
    assert cycle_record["errors"] == ["browser update failed"]
    assert cycle_record["history"]["previous_latest_draw"] == draws[-2].draw_number
    assert cycle_record["history"]["new_latest_draw"] == draws[-2].draw_number
    assert cycle_record["history"]["appended"] == 0
    assert cycle_record["history"]["update_status"] == HISTORY_UPDATE_SOURCE_FAILURE


def test_cycle_conflicting_draw_failure(tmp_path: Path) -> None:
    history = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=900, count=110, start_date=date(2024, 1, 2))
    _write_history(history, draws)
    conflict = HistoricalDraw(
        MINI_LOTO,
        draws[-1].draw_number,
        draws[-1].draw_date,
        tuple(range(1, MINI_LOTO.numbers_per_ticket + 1)),
        (MINI_LOTO.numbers_per_ticket + 1,),
    )

    with pytest.raises(ResearchValidationError, match="conflicting historical record"):
        run_post_draw_cycle(
            MINI_LOTO,
            _config(),
            history_path=history,
            prediction_root=tmp_path / "predictions",
            settlement_root=tmp_path / "settlements",
            tickets_per_draw=3,
            history_updater=_FakeUpdater(history, (conflict,)),
        )


@pytest.mark.parametrize("ticket_count", [1, 2, 3, 5])
def test_cycle_supports_configurable_ticket_counts(ticket_count: int, tmp_path: Path) -> None:
    history = tmp_path / "mini.csv"
    draws = _draws(MINI_LOTO, start_number=900, count=110, start_date=date(2024, 1, 2))
    _write_history(history, draws)

    result = run_post_draw_cycle(
        MINI_LOTO,
        _config(),
        history_path=history,
        prediction_root=tmp_path / "predictions",
        settlement_root=tmp_path / "settlements",
        tickets_per_draw=ticket_count,
        history_updater=_FakeUpdater(history, ()),
    )

    assert result.next_prediction.tickets == ticket_count
