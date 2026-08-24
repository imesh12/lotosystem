from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6
from backend.app.research.browser_mizuho import BrowserMizuhoCollector
from backend.app.research.collectors import CollectorInterface
from backend.app.research.data import HistoricalDraw, load_draws_csv, validate_draw_sequence
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import (
    DEFAULT_HISTORY_START,
    LocalMizuhoArchiveCollector,
    MizuhoHistoricalCollector,
)

CANONICAL_HISTORY_COLUMNS = (
    "lottery",
    "draw_number",
    "draw_date",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "bonus",
    "source",
    "source_url",
    "retrieved_at",
    "content_hash",
)
HISTORY_UPDATE_NEW_RESULT = "NEW_RESULT"
HISTORY_UPDATE_NO_NEW_RESULT = "NO_NEW_RESULT"
HISTORY_UPDATE_SOURCE_FAILURE = "SOURCE_FAILURE"
RESULT_SOURCE_MIZUHO = "mizuho"
RESULT_SOURCE_SECONDARY = "secondary"
RESULT_SOURCE_MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ProvenanceImportSummary:
    sources: tuple[str, ...]
    source_urls: tuple[str, ...]
    content_hashes: tuple[str, ...]
    records_with_retrieved_at: int


@dataclass(frozen=True, slots=True)
class HistoryVerification:
    lottery: str
    first_draw_date: str | None
    last_draw_date: str | None
    first_draw_number: int | None
    last_draw_number: int | None
    draw_count: int
    validation_errors: tuple[str, ...]
    duplicate_draw_numbers: tuple[int, ...]
    duplicate_records: tuple[str, ...]
    missing_draw_numbers: tuple[int, ...]
    dataset_hash: str | None
    provenance: ProvenanceImportSummary


@dataclass(frozen=True, slots=True)
class HistorySourceAttempt:
    source: str
    result: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HistoryUpdateResult:
    output_path: str
    fetched_count: int
    existing_count: int
    written_count: int
    appended_count: int
    unchanged_count: int
    verification: HistoryVerification
    update_status: str = HISTORY_UPDATE_NEW_RESULT
    source_attempts: tuple[HistorySourceAttempt, ...] = ()
    selected_source: str | None = None
    fallback_used: bool = False
    warnings: tuple[str, ...] = ()


def canonical_history_path(lottery: LotteryDefinition) -> Path:
    filename = "loto6_history.csv" if lottery.code == LOTO6.code else "mini_loto_history.csv"
    return Path("data") / "processed" / filename


def update_mizuho_history(
    lottery: LotteryDefinition,
    *,
    output_path: str | Path | None = None,
    source_dir: str | Path | None = None,
    start_date: date = DEFAULT_HISTORY_START,
    end_date: date | None = None,
    incremental: bool = True,
) -> HistoryUpdateResult:
    destination = Path(output_path) if output_path else canonical_history_path(lottery)
    existing = _load_existing(destination, lottery)
    minimum_draw_number = _incremental_minimum_draw_number(existing) if incremental else None
    if source_dir:
        collector = LocalMizuhoArchiveCollector(
            source_dir,
            start_date=start_date,
            end_date=end_date,
            minimum_draw_number=minimum_draw_number,
        )
    else:
        collector = MizuhoHistoricalCollector(
            start_date=start_date,
            end_date=end_date,
            minimum_draw_number=minimum_draw_number,
        )
    return update_history(destination, lottery, collector, existing_draws=existing)


def update_history_with_sources(
    lottery: LotteryDefinition,
    *,
    output_path: str | Path | None = None,
    start_date: date = DEFAULT_HISTORY_START,
    end_date: date | None = None,
    incremental: bool = True,
    headed: bool = False,
    row_timeout_ms: int = 7_000,
    source_order: tuple[str, ...] = (RESULT_SOURCE_MIZUHO, RESULT_SOURCE_SECONDARY),
) -> HistoryUpdateResult:
    from backend.app.research.result_sources import SMBCResultSource

    destination = Path(output_path) if output_path else canonical_history_path(lottery)
    existing = _load_existing(destination, lottery)
    minimum_draw_number = _incremental_minimum_draw_number(existing) if incremental else None
    attempts: list[HistorySourceAttempt] = []

    for source_name in source_order:
        if source_name == RESULT_SOURCE_MIZUHO:
            collector: CollectorInterface = BrowserMizuhoCollector(
                start_date=start_date,
                end_date=end_date,
                minimum_draw_number=minimum_draw_number,
                headed=headed,
                row_timeout_ms=row_timeout_ms,
            )
        elif source_name == RESULT_SOURCE_SECONDARY:
            collector = SMBCResultSource(
                start_date=start_date,
                end_date=end_date,
                minimum_draw_number=minimum_draw_number,
            )
        else:
            raise ResearchValidationError(f"unknown result source: {source_name}")

        try:
            result = update_history(destination, lottery, collector, existing_draws=existing)
        except ResearchValidationError as exc:
            attempts.append(
                HistorySourceAttempt(
                    source=source_name,
                    result=HISTORY_UPDATE_SOURCE_FAILURE,
                    error=str(exc),
                )
            )
            continue

        attempts.append(
            HistorySourceAttempt(source=source_name, result=result.update_status, error=None)
        )
        return _with_source_metadata(
            result,
            attempts=tuple(attempts),
            selected_source=source_name,
            fallback_used=len(attempts) > 1,
        )

    errors = "; ".join(f"{attempt.source}: {attempt.error}" for attempt in attempts)
    raise ResearchValidationError(
        f"all automated result sources failed; manual result entry required: {errors}"
    )


def bootstrap_mizuho_history_with_browser(
    lottery: LotteryDefinition,
    *,
    output_path: str | Path | None = None,
    start_date: date = DEFAULT_HISTORY_START,
    end_date: date | None = None,
    headed: bool = False,
    row_timeout_ms: int = 7_000,
    incremental: bool = True,
) -> HistoryUpdateResult:
    destination = Path(output_path) if output_path else canonical_history_path(lottery)
    existing = _load_existing(destination, lottery)
    minimum_draw_number = _incremental_minimum_draw_number(existing) if incremental else None
    collector = BrowserMizuhoCollector(
        start_date=start_date,
        end_date=end_date,
        minimum_draw_number=minimum_draw_number,
        headed=headed,
        row_timeout_ms=row_timeout_ms,
    )
    return update_history(destination, lottery, collector, existing_draws=existing)


def append_manual_result(
    lottery: LotteryDefinition,
    *,
    draw_number: int,
    draw_date: date,
    main_numbers: tuple[int, ...],
    bonus_numbers: tuple[int, ...],
    output_path: str | Path | None = None,
    confirmed: bool = False,
) -> HistoryUpdateResult:
    if not confirmed:
        raise ResearchValidationError("manual result entry requires --confirm-manual")
    destination = Path(output_path) if output_path else canonical_history_path(lottery)
    existing = _load_existing(destination, lottery)
    now = datetime.now(UTC)
    draw = HistoricalDraw(
        lottery=lottery,
        draw_number=draw_number,
        draw_date=draw_date,
        main_numbers=main_numbers,
        bonus_numbers=bonus_numbers,
        source=RESULT_SOURCE_MANUAL,
        source_url="manual:cli",
        retrieved_at=now,
        content_hash=f"manual:{lottery.code}:{draw_number}:{draw_date.isoformat()}",
    )
    result = update_history(
        destination,
        lottery,
        _StaticDrawCollector((draw,)),
        existing_draws=existing,
    )
    return _with_source_metadata(
        result,
        attempts=(
            HistorySourceAttempt(
                source=RESULT_SOURCE_MANUAL,
                result=result.update_status,
                error=None,
            ),
        ),
        selected_source=RESULT_SOURCE_MANUAL,
        fallback_used=False,
    )


def update_history(
    output_path: str | Path,
    lottery: LotteryDefinition,
    collector: CollectorInterface,
    *,
    existing_draws: tuple[HistoricalDraw, ...] | None = None,
) -> HistoryUpdateResult:
    destination = Path(output_path)
    existing = (
        existing_draws if existing_draws is not None else _load_existing(destination, lottery)
    )
    fetched = collector.collect(lottery)
    merged, appended_count, unchanged_count = merge_historical_draws(existing, fetched)
    verification = verify_history(merged, lottery)
    if verification.validation_errors:
        raise ResearchValidationError("; ".join(verification.validation_errors))
    write_canonical_history_csv(merged, destination)
    return HistoryUpdateResult(
        output_path=str(destination),
        fetched_count=len(fetched),
        existing_count=len(existing),
        written_count=len(merged),
        appended_count=appended_count,
        unchanged_count=unchanged_count,
        verification=verification,
        update_status=HISTORY_UPDATE_NEW_RESULT if appended_count else HISTORY_UPDATE_NO_NEW_RESULT,
    )


class _StaticDrawCollector(CollectorInterface):
    def __init__(self, draws: tuple[HistoricalDraw, ...]) -> None:
        self.draws = draws

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        return tuple(draw for draw in self.draws if draw.lottery.code == lottery.code)


def _with_source_metadata(
    result: HistoryUpdateResult,
    *,
    attempts: tuple[HistorySourceAttempt, ...],
    selected_source: str | None,
    fallback_used: bool,
) -> HistoryUpdateResult:
    return HistoryUpdateResult(
        output_path=result.output_path,
        fetched_count=result.fetched_count,
        existing_count=result.existing_count,
        written_count=result.written_count,
        appended_count=result.appended_count,
        unchanged_count=result.unchanged_count,
        verification=result.verification,
        update_status=result.update_status,
        source_attempts=attempts,
        selected_source=selected_source,
        fallback_used=fallback_used,
        warnings=result.warnings,
    )


def merge_historical_draws(
    existing_draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    fetched_draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
) -> tuple[tuple[HistoricalDraw, ...], int, int]:
    merged: dict[tuple[str, int], HistoricalDraw] = {}
    for draw in existing_draws:
        merged[(str(draw.lottery.code), draw.draw_number)] = draw

    appended_count = 0
    unchanged_count = 0
    for draw in fetched_draws:
        key = (str(draw.lottery.code), draw.draw_number)
        existing = merged.get(key)
        if existing is None:
            merged[key] = draw
            appended_count += 1
            continue
        if existing.canonical_identity != draw.canonical_identity:
            raise ResearchValidationError(
                f"conflicting historical record for {key[0]} #{key[1]}: "
                "existing canonical data differs from fetched source"
            )
        unchanged_count += 1

    ordered = validate_draw_sequence(
        tuple(sorted(merged.values(), key=lambda draw: (draw.draw_date, draw.draw_number)))
    )
    return ordered, appended_count, unchanged_count


def verify_history(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> HistoryVerification:
    validation_errors: list[str] = []
    ordered: tuple[HistoricalDraw, ...] = ()
    try:
        ordered = validate_lottery_dataset(draws, lottery)
    except ResearchValidationError as exc:
        validation_errors.append(str(exc))
        ordered = tuple(sorted(draws, key=lambda draw: (draw.draw_date, draw.draw_number)))

    draw_number_counts = Counter(draw.draw_number for draw in draws)
    identity_counts = Counter(draw.canonical_identity for draw in draws)
    duplicate_draw_numbers = tuple(
        sorted(draw_number for draw_number, count in draw_number_counts.items() if count > 1)
    )
    duplicate_records = tuple(
        sorted(identity for identity, count in identity_counts.items() if count > 1)
    )

    return HistoryVerification(
        lottery=str(lottery.code),
        first_draw_date=ordered[0].draw_date.isoformat() if ordered else None,
        last_draw_date=ordered[-1].draw_date.isoformat() if ordered else None,
        first_draw_number=ordered[0].draw_number if ordered else None,
        last_draw_number=ordered[-1].draw_number if ordered else None,
        draw_count=len(ordered),
        validation_errors=tuple(validation_errors),
        duplicate_draw_numbers=duplicate_draw_numbers,
        duplicate_records=duplicate_records,
        missing_draw_numbers=find_missing_draw_numbers(ordered),
        dataset_hash=calculate_dataset_hash(ordered) if ordered and not validation_errors else None,
        provenance=_provenance_summary(ordered),
    )


def write_canonical_history_csv(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    path: str | Path,
) -> None:
    ordered = validate_draw_sequence(
        tuple(sorted(draws, key=lambda draw: (draw.draw_date, draw.draw_number)))
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANONICAL_HISTORY_COLUMNS)
        writer.writeheader()
        for draw in ordered:
            writer.writerow(_canonical_row(draw))


def _canonical_row(draw: HistoricalDraw) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "lottery": str(draw.lottery.code),
        "draw_number": draw.draw_number,
        "draw_date": draw.draw_date.isoformat(),
        "source": draw.source or "",
        "source_url": draw.source_url or "",
        "retrieved_at": draw.retrieved_at.isoformat() if draw.retrieved_at else "",
        "content_hash": draw.content_hash or "",
        "bonus": draw.bonus_numbers[0] if draw.bonus_numbers else "",
    }
    for index in range(1, 7):
        row[f"n{index}"] = draw.main_numbers[index - 1] if index <= len(draw.main_numbers) else ""
    return row


def _load_existing(path: Path, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
    if not path.exists():
        return ()
    return load_draws_csv(path, lottery)


def _incremental_minimum_draw_number(existing: tuple[HistoricalDraw, ...]) -> int | None:
    if not existing:
        return None
    missing_draw_numbers = find_missing_draw_numbers(existing)
    if missing_draw_numbers:
        return max(1, missing_draw_numbers[0] - 1)
    return max(1, existing[-1].draw_number - 1)


def find_missing_draw_numbers(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
) -> tuple[int, ...]:
    if not draws:
        return ()
    draw_numbers = {draw.draw_number for draw in draws}
    first = min(draw_numbers)
    last = max(draw_numbers)
    return tuple(number for number in range(first, last + 1) if number not in draw_numbers)


def _provenance_summary(draws: tuple[HistoricalDraw, ...]) -> ProvenanceImportSummary:
    return ProvenanceImportSummary(
        sources=tuple(sorted({draw.source for draw in draws if draw.source})),
        source_urls=tuple(sorted({draw.source_url for draw in draws if draw.source_url})),
        content_hashes=tuple(sorted({draw.content_hash for draw in draws if draw.content_hash})),
        records_with_retrieved_at=sum(1 for draw in draws if draw.retrieved_at is not None),
    )
