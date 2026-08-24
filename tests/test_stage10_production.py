from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.production import (
    PREDICTION_STATUS_EVALUATED,
    PREDICTION_STATUS_PENDING,
    evaluate_pending_predictions,
    generate_next_prediction,
    load_prediction_record,
    next_scheduled_draw_date,
    save_prediction_record,
)


def _loto6_draws(count: int = 110) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 31) + 1
        main = tuple(sorted(((start + offset * 5 - 1) % 43) + 1 for offset in range(6)))
        bonus = next(number for number in range(1, 44) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=LOTO6,
                draw_number=1000 + index,
                draw_date=date(2024, 1, 1) + timedelta(days=index * 3),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _mini_draws(count: int = 110) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 20) + 1
        main = tuple(sorted(((start + offset * 4 - 1) % 31) + 1 for offset in range(5)))
        bonus = next(number for number in range(1, 32) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=MINI_LOTO,
                draw_number=500 + index,
                draw_date=date(2024, 1, 2) + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _config() -> ResearchConfig:
    return ResearchConfig(seed=123456)


def _write_csv(path: Path, draws: tuple[HistoricalDraw, ...]) -> None:
    rows = ["lottery,draw_number,draw_date,main_numbers,bonus_numbers"]
    for draw in draws:
        rows.append(
            ",".join(
                (
                    str(draw.lottery.code),
                    str(draw.draw_number),
                    draw.draw_date.isoformat(),
                    " ".join(str(number) for number in draw.main_numbers),
                    " ".join(str(number) for number in draw.bonus_numbers),
                )
            )
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_next_target_draw_detection() -> None:
    assert next_scheduled_draw_date(date(2026, 8, 20), LOTO6) == date(2026, 8, 24)
    assert next_scheduled_draw_date(date(2026, 8, 18), MINI_LOTO) == date(2026, 8, 25)


@pytest.mark.parametrize("ticket_count", [1, 2, 3, 5])
def test_generate_configurable_ticket_counts(ticket_count: int, tmp_path: Path) -> None:
    result = generate_next_prediction(
        _loto6_draws(),
        LOTO6,
        _config(),
        tickets_per_draw=ticket_count,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    assert result.record.status == PREDICTION_STATUS_PENDING
    assert len(result.record.tickets) == ticket_count
    assert len({ticket.numbers for ticket in result.record.tickets}) == ticket_count
    assert result.record.cost_yen == ticket_count * LOTO6.ticket_price_yen
    assert result.record.gross_winnings_yen is None


def test_mini_loto_generation_is_valid_and_reproducible(tmp_path: Path) -> None:
    first = generate_next_prediction(
        _mini_draws(),
        MINI_LOTO,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    second = generate_next_prediction(
        _mini_draws(),
        MINI_LOTO,
        _config(),
        tickets_per_draw=5,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert second.existing_record is True
    assert second.record.tickets == first.record.tickets
    assert all(
        len(ticket.numbers) == MINI_LOTO.numbers_per_ticket for ticket in first.record.tickets
    )


def test_immutable_evaluated_record_and_idempotent_evaluation(tmp_path: Path) -> None:
    draws = _loto6_draws()
    generated = generate_next_prediction(
        draws[:-1],
        LOTO6,
        _config(),
        tickets_per_draw=2,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    first_eval = evaluate_pending_predictions(draws, LOTO6, prediction_root=tmp_path)
    second_eval = evaluate_pending_predictions(draws, LOTO6, prediction_root=tmp_path)
    record = load_prediction_record(generated.record_path)

    assert first_eval.evaluated_count == 1
    assert second_eval.evaluated_count == 0
    assert record.status == PREDICTION_STATUS_EVALUATED
    assert record.evaluation is not None
    assert record.evaluation["gross_winnings_yen"] is None
    assert record.evaluation["best_match_count"] >= 0
    with pytest.raises(ResearchValidationError):
        save_prediction_record(record, generated.record_path)


def test_future_data_does_not_mutate_existing_prediction(tmp_path: Path) -> None:
    draws = _loto6_draws()
    generated = generate_next_prediction(
        draws[:-1],
        LOTO6,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    mutated_future = list(draws)
    future = mutated_future[-1]
    mutated_future[-1] = HistoricalDraw(
        LOTO6,
        future.draw_number,
        future.draw_date,
        tuple(reversed(future.main_numbers)),
        future.bonus_numbers,
    )
    repeated = generate_next_prediction(
        tuple(mutated_future[:-1]),
        LOTO6,
        _config(),
        tickets_per_draw=3,
        prediction_root=tmp_path,
        generated_at=datetime(2026, 8, 25, tzinfo=UTC),
    )

    assert repeated.existing_record is True
    assert repeated.record.tickets == generated.record.tickets


def test_generate_next_cli_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini.csv"
    _write_csv(csv_path, _mini_draws())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "MINI_LOTO",
            "--data",
            str(csv_path),
            "--seed",
            "123456",
            "--tickets-per-draw",
            "3",
            "--prediction-root",
            str(tmp_path / "predictions"),
            "generate-next",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == PREDICTION_STATUS_PENDING
    assert len(payload["tickets"]) == 3
