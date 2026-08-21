from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import get_lottery_definition
from backend.app.research.exceptions import ResearchValidationError


@dataclass(frozen=True, slots=True)
class HistoricalDraw:
    lottery: LotteryDefinition
    draw_number: int
    draw_date: date
    main_numbers: tuple[int, ...]
    bonus_numbers: tuple[int, ...] = ()
    source: str | None = None
    source_url: str | None = None
    retrieved_at: datetime | None = None
    content_hash: str | None = None
    source_row: int | None = None

    def __post_init__(self) -> None:
        if self.draw_number <= 0:
            raise ResearchValidationError("draw_number must be positive")
        try:
            main_numbers = self.lottery.validate_main_numbers(self.main_numbers)
            bonus_numbers = self.lottery.validate_bonus_numbers(
                self.bonus_numbers,
                main_numbers=main_numbers,
            )
        except ValueError as exc:
            raise ResearchValidationError(str(exc)) from exc
        object.__setattr__(self, "main_numbers", main_numbers)
        object.__setattr__(self, "bonus_numbers", bonus_numbers)

    @property
    def canonical_main_numbers(self) -> str:
        return self.lottery.canonical_ticket(self.main_numbers)

    @property
    def canonical_identity(self) -> str:
        return "|".join(
            (
                str(self.lottery.code),
                str(self.draw_number),
                self.draw_date.isoformat(),
                ",".join(str(number) for number in self.main_numbers),
                ",".join(str(number) for number in self.bonus_numbers),
            )
        )


def validate_draw_sequence(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
) -> tuple[HistoricalDraw, ...]:
    if not draws:
        return ()

    for index, (previous, current) in enumerate(zip(draws, draws[1:], strict=False), start=2):
        if (current.draw_date, current.draw_number) < (previous.draw_date, previous.draw_number):
            raise ResearchValidationError(f"row {index}: source rows are not chronological")

    ordered = tuple(sorted(draws, key=lambda draw: (draw.draw_date, draw.draw_number)))
    seen_draw_numbers: set[int] = set()
    seen_identities: set[tuple[str, int]] = set()

    for draw in ordered:
        identity = (str(draw.lottery.code), draw.draw_number)
        if identity in seen_identities:
            raise ResearchValidationError(f"duplicate draw identity: {identity[0]} #{identity[1]}")
        seen_identities.add(identity)
        if draw.draw_number in seen_draw_numbers:
            raise ResearchValidationError(f"duplicate draw_number: {draw.draw_number}")
        seen_draw_numbers.add(draw.draw_number)

    return ordered


def load_draws_csv(path: str | Path, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=2):
            try:
                row_lottery = _parse_lottery(row, lottery, row_number)
                rows.append(
                    HistoricalDraw(
                        lottery=row_lottery,
                        draw_number=_parse_draw_number(row, row_number),
                        draw_date=_parse_draw_date(row, row_number),
                        main_numbers=_parse_numbers(
                            row, row_lottery.numbers_per_ticket, "main_numbers", "n", row_number
                        ),
                        bonus_numbers=_parse_numbers(
                            row, row_lottery.bonus_numbers, "bonus_numbers", "bonus", row_number
                        ),
                        source=row.get("source") or None,
                        source_url=row.get("source_url") or None,
                        retrieved_at=_parse_optional_datetime(row.get("retrieved_at"), row_number),
                        content_hash=row.get("content_hash") or None,
                        source_row=row_number,
                    )
                )
            except (KeyError, ValueError, ResearchValidationError) as exc:
                raise ResearchValidationError(f"row {row_number}: {exc}") from exc
    return validate_draw_sequence(rows)


def _parse_lottery(
    row: dict[str, str],
    expected_lottery: LotteryDefinition,
    row_number: int,
) -> LotteryDefinition:
    raw_lottery = (row.get("lottery") or "").strip()
    if not raw_lottery:
        return expected_lottery
    try:
        row_lottery = get_lottery_definition(raw_lottery)
    except ValueError as exc:
        raise ResearchValidationError(f"invalid lottery code {raw_lottery!r}") from exc
    if row_lottery.code != expected_lottery.code:
        raise ResearchValidationError(
            f"field lottery: expected {expected_lottery.code}, found {row_lottery.code}"
        )
    return row_lottery


def _parse_draw_number(row: dict[str, str], row_number: int) -> int:
    raw_value = (row.get("draw_number") or "").strip()
    if not raw_value:
        raise ResearchValidationError("field draw_number is required")
    try:
        draw_number = int(raw_value)
    except ValueError as exc:
        raise ResearchValidationError(
            f"field draw_number is not an integer: {raw_value!r}"
        ) from exc
    if draw_number <= 0:
        raise ResearchValidationError("field draw_number must be positive")
    return draw_number


def _parse_draw_date(row: dict[str, str], row_number: int) -> date:
    raw_value = (row.get("draw_date") or "").strip()
    if not raw_value:
        raise ResearchValidationError("field draw_date is required")
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ResearchValidationError(f"field draw_date is malformed: {raw_value!r}") from exc


def _parse_optional_datetime(raw_value: str | None, row_number: int) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise ResearchValidationError(f"field retrieved_at is malformed: {raw_value!r}") from exc


def _parse_numbers(
    row: dict[str, str],
    expected_count: int,
    packed_field: str,
    column_prefix: str,
    row_number: int,
) -> tuple[int, ...]:
    if expected_count == 0:
        return ()

    packed_value = row.get(packed_field, "").strip()
    if packed_value:
        separators = str.maketrans({",": " ", "-": " ", "|": " "})
        values = packed_value.translate(separators).split()
        return _parse_number_values(values, expected_count, packed_field)

    values: list[int] = []
    for index in range(1, expected_count + 1):
        raw_value = row.get(f"{column_prefix}{index}", "").strip()
        if raw_value:
            values.append(raw_value)

    if column_prefix == "bonus" and not values:
        raw_value = row.get("bonus", "").strip()
        if raw_value:
            values.append(raw_value)

    return _parse_number_values(values, expected_count, column_prefix)


def _parse_number_values(
    values: list[str], expected_count: int, field_name: str
) -> tuple[int, ...]:
    if len(values) != expected_count:
        raise ResearchValidationError(
            f"field {field_name}: expected {expected_count} numbers, found {len(values)}"
        )
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise ResearchValidationError(f"field {field_name}: invalid number {value!r}") from exc
    return tuple(parsed)
