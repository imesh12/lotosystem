from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from backend.app.domain.rules import MINI_LOTO
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import SourceDocument
from backend.app.research.result_sources import SMBC_MINI_LOTO_XML_URL, SMBC_SOURCE
from backend.app.research.settlement import SETTLEMENT_ROOT, load_settlement, settlement_lottery_dir
from backend.app.research.stage28_ticket_popularity import (
    PopularityScore,
    WinnerCountObservation,
    score_ticket,
)

STAGE28B_SCHEMA_VERSION = "v2-stage28b-mini-loto-economic-history-v1"
STAGE28B_OUTPUT_PATH = Path("data") / "exports" / "stage28" / "mini_loto_economic_history.csv"
SOURCE_SETTLEMENT = "local_settlement"
SOURCE_PRIORITY = {
    "mizuho_bank": 1,
    SMBC_SOURCE: 2,
    SOURCE_SETTLEMENT: 3,
    "manual": 4,
}

ECONOMIC_FIELDNAMES = (
    "lottery",
    "draw_number",
    "draw_date",
    "sales_amount_yen",
    "first_prize_winners",
    "first_prize_payout_yen",
    "second_prize_winners",
    "second_prize_payout_yen",
    "third_prize_winners",
    "third_prize_payout_yen",
    "fourth_prize_winners",
    "fourth_prize_payout_yen",
    "source",
    "source_url",
    "fetched_at",
    "source_quality",
)


@dataclass(frozen=True, slots=True)
class MiniLotoEconomicResult:
    lottery: str
    draw_number: int
    draw_date: str
    sales_amount_yen: int | None
    first_prize_winners: int | None
    first_prize_payout_yen: int | None
    second_prize_winners: int | None
    second_prize_payout_yen: int | None
    third_prize_winners: int | None
    third_prize_payout_yen: int | None
    fourth_prize_winners: int | None
    fourth_prize_payout_yen: int | None
    source: str
    source_url: str | None
    fetched_at: str | None
    source_quality: str


@dataclass(frozen=True, slots=True)
class EconomicCoverageReport:
    schema_version: str
    earliest_draw: int | None
    earliest_date: str | None
    latest_draw: int | None
    latest_date: str | None
    total_rows: int
    rows_with_first_prize_winners: int
    rows_with_first_prize_payout: int
    rows_with_sales_amount: int
    usable_stage28_observations: int
    coverage_percentage: float
    missing_ranges: tuple[tuple[int, int], ...]
    source_breakdown: dict[str, int]
    conflicts: tuple[str, ...]


def build_mini_loto_economic_history(
    draws: tuple[HistoricalDraw, ...],
    *,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    smbc_xml_text: str | None = None,
    fetched_at: datetime | None = None,
) -> tuple[MiniLotoEconomicResult, ...]:
    by_draw = {draw.draw_number: draw for draw in draws}
    rows: list[MiniLotoEconomicResult] = []
    rows.extend(_rows_from_settlements(by_draw, settlement_root))
    if smbc_xml_text:
        document = SourceDocument(
            url=SMBC_MINI_LOTO_XML_URL,
            text=smbc_xml_text,
            retrieved_at=fetched_at or datetime.now(UTC),
            content_hash="provided-test-xml",
        )
        rows.extend(parse_smbc_mini_loto_economic_xml(smbc_xml_text, by_draw, document))
    return merge_economic_results(tuple(rows))


def parse_smbc_mini_loto_economic_xml(
    xml_text: str,
    draws_by_number: dict[int, HistoricalDraw],
    document: SourceDocument,
) -> tuple[MiniLotoEconomicResult, ...]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ResearchValidationError(f"invalid SMBC economic XML: {exc}") from exc
    rows: list[MiniLotoEconomicResult] = []
    for element in root.findall("data"):
        attrs = element.attrib
        if attrs.get("GAME_TYPE") != "04":
            continue
        draw_number = parse_count(attrs.get("KAIGOU"))
        if draw_number is None or draw_number not in draws_by_number:
            continue
        draw = draws_by_number[draw_number]
        rows.append(
            MiniLotoEconomicResult(
                lottery=str(MINI_LOTO.code),
                draw_number=draw_number,
                draw_date=draw.draw_date.isoformat(),
                sales_amount_yen=_first_int_attr(
                    attrs,
                    ("URIAGE_KINGAKU", "URIAGE", "SALES_AMOUNT", "SALES_AMOUNT_YEN"),
                ),
                first_prize_winners=parse_count(attrs.get("TOUSEN_KUTI1")),
                first_prize_payout_yen=parse_yen(attrs.get("TOUSEN_KINGAKU1")),
                second_prize_winners=parse_count(attrs.get("TOUSEN_KUTI2")),
                second_prize_payout_yen=parse_yen(attrs.get("TOUSEN_KINGAKU2")),
                third_prize_winners=parse_count(attrs.get("TOUSEN_KUTI3")),
                third_prize_payout_yen=parse_yen(attrs.get("TOUSEN_KINGAKU3")),
                fourth_prize_winners=parse_count(attrs.get("TOUSEN_KUTI4")),
                fourth_prize_payout_yen=parse_yen(attrs.get("TOUSEN_KINGAKU4")),
                source=SMBC_SOURCE,
                source_url=document.url,
                fetched_at=document.retrieved_at.astimezone(UTC).isoformat(),
                source_quality="official_secondary_public_xml",
            )
        )
    return tuple(rows)


def merge_economic_results(
    rows: tuple[MiniLotoEconomicResult, ...],
) -> tuple[MiniLotoEconomicResult, ...]:
    merged: dict[int, MiniLotoEconomicResult] = {}
    for row in rows:
        validate_economic_result(row)
        current = merged.get(row.draw_number)
        if current is None:
            merged[row.draw_number] = row
            continue
        if _scientific_values(current) != _scientific_values(row):
            if _source_rank(row.source) == _source_rank(current.source):
                raise ResearchValidationError(
                    f"conflicting Mini Loto economic record for draw #{row.draw_number}"
                )
            preferred, secondary = (
                (row, current)
                if _source_rank(row.source) < _source_rank(current.source)
                else (current, row)
            )
            if not _compatible_missing_fill(preferred, secondary):
                raise ResearchValidationError(
                    f"conflicting Mini Loto economic record for draw #{row.draw_number}"
                )
            merged[row.draw_number] = _fill_missing(preferred, secondary)
        else:
            merged[row.draw_number] = current
    return tuple(merged[key] for key in sorted(merged))


def validate_economic_result(row: MiniLotoEconomicResult) -> MiniLotoEconomicResult:
    if row.lottery != str(MINI_LOTO.code):
        raise ResearchValidationError("Mini Loto economic row has invalid lottery")
    if row.draw_number <= 0:
        raise ResearchValidationError("Mini Loto economic row draw_number must be positive")
    date.fromisoformat(row.draw_date)
    for field in ECONOMIC_FIELDNAMES:
        if field.endswith("_yen") or field.endswith("_winners"):
            value = getattr(row, field)
            if value is not None and value < 0:
                raise ResearchValidationError(f"{field} must be >= 0")
    return row


def write_economic_history_csv(
    rows: tuple[MiniLotoEconomicResult, ...],
    path: str | Path = STAGE28B_OUTPUT_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=ECONOMIC_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))
    return destination


def load_economic_history_csv(path: str | Path) -> tuple[MiniLotoEconomicResult, ...]:
    rows: list[MiniLotoEconomicResult] = []
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            rows.append(
                MiniLotoEconomicResult(
                    lottery=row["lottery"],
                    draw_number=int(row["draw_number"]),
                    draw_date=row["draw_date"],
                    sales_amount_yen=parse_yen(row.get("sales_amount_yen")),
                    first_prize_winners=parse_count(row.get("first_prize_winners")),
                    first_prize_payout_yen=parse_yen(row.get("first_prize_payout_yen")),
                    second_prize_winners=parse_count(row.get("second_prize_winners")),
                    second_prize_payout_yen=parse_yen(row.get("second_prize_payout_yen")),
                    third_prize_winners=parse_count(row.get("third_prize_winners")),
                    third_prize_payout_yen=parse_yen(row.get("third_prize_payout_yen")),
                    fourth_prize_winners=parse_count(row.get("fourth_prize_winners")),
                    fourth_prize_payout_yen=parse_yen(row.get("fourth_prize_payout_yen")),
                    source=row["source"],
                    source_url=row.get("source_url") or None,
                    fetched_at=row.get("fetched_at") or None,
                    source_quality=row["source_quality"],
                )
            )
    return merge_economic_results(tuple(rows))


def coverage_report(
    rows: tuple[MiniLotoEconomicResult, ...],
    canonical_draws: tuple[HistoricalDraw, ...],
) -> EconomicCoverageReport:
    row_draws = {row.draw_number for row in rows}
    sources = Counter(row.source for row in rows)
    return EconomicCoverageReport(
        schema_version=STAGE28B_SCHEMA_VERSION,
        earliest_draw=rows[0].draw_number if rows else None,
        earliest_date=rows[0].draw_date if rows else None,
        latest_draw=rows[-1].draw_number if rows else None,
        latest_date=rows[-1].draw_date if rows else None,
        total_rows=len(rows),
        rows_with_first_prize_winners=sum(row.first_prize_winners is not None for row in rows),
        rows_with_first_prize_payout=sum(row.first_prize_payout_yen is not None for row in rows),
        rows_with_sales_amount=sum(row.sales_amount_yen is not None for row in rows),
        usable_stage28_observations=sum(_usable_for_stage28(row) for row in rows),
        coverage_percentage=(len(rows) / len(canonical_draws) * 100) if canonical_draws else 0.0,
        missing_ranges=_missing_ranges(
            tuple(draw.draw_number for draw in canonical_draws), row_draws
        ),
        source_breakdown=dict(sorted(sources.items())),
        conflicts=(),
    )


def economic_rows_to_stage28_observations(
    rows: tuple[MiniLotoEconomicResult, ...],
    draws_by_number: dict[int, HistoricalDraw],
    universe_scores: tuple[PopularityScore, ...] | None = None,
) -> tuple[WinnerCountObservation, ...]:
    score_by_ticket = {score.ticket: score.normalized_score for score in universe_scores or ()}
    observations: list[WinnerCountObservation] = []
    for row in rows:
        draw = draws_by_number.get(row.draw_number)
        if draw is None:
            raise ResearchValidationError(f"missing Mini Loto canonical draw #{row.draw_number}")
        ticket = draw.main_numbers
        score = score_by_ticket.get(ticket, score_ticket(ticket).normalized_score)
        tickets_sold = (
            row.sales_amount_yen / MINI_LOTO.ticket_price_yen
            if row.sales_amount_yen is not None
            else None
        )
        normalized = (
            row.first_prize_winners / tickets_sold
            if row.first_prize_winners is not None and tickets_sold
            else None
        )
        observations.append(
            WinnerCountObservation(
                draw_number=row.draw_number,
                draw_date=row.draw_date,
                main_numbers=ticket,
                popularity_score=score,
                first_prize_winners=row.first_prize_winners,
                first_prize_payout_yen=row.first_prize_payout_yen,
                sales_amount_yen=row.sales_amount_yen,
                estimated_tickets_sold=tickets_sold,
                normalized_winner_rate=normalized,
            )
        )
    return tuple(observations)


def acquire_stage28b_economic_history(
    *,
    history_path: str | Path = Path("data") / "processed" / "mini_loto_history.csv",
    settlement_root: str | Path = SETTLEMENT_ROOT,
    output_path: str | Path = STAGE28B_OUTPUT_PATH,
) -> tuple[tuple[MiniLotoEconomicResult, ...], EconomicCoverageReport, Path]:
    draws = load_draws_csv(history_path, MINI_LOTO)
    rows = build_mini_loto_economic_history(draws, settlement_root=settlement_root)
    path = write_economic_history_csv(rows, output_path)
    return rows, coverage_report(rows, draws), path


def parse_yen(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    cleaned = value.strip().replace(",", "").replace("円", "").replace("¥", "").replace("￥", "")
    if not cleaned:
        return None
    if not cleaned.isdigit():
        raise ResearchValidationError(f"invalid yen value: {value!r}")
    return int(cleaned)


def parse_count(value: str | int | None) -> int | None:
    if isinstance(value, str):
        value = value.strip().replace("口", "").replace("件", "")
    parsed = parse_yen(value)
    if parsed is None:
        return None
    return parsed


def _rows_from_settlements(
    draws_by_number: dict[int, HistoricalDraw],
    settlement_root: str | Path,
) -> tuple[MiniLotoEconomicResult, ...]:
    rows: list[MiniLotoEconomicResult] = []
    for path in sorted(settlement_lottery_dir(settlement_root, MINI_LOTO).glob("*.json")):
        settlement = load_settlement(path)
        draw = draws_by_number.get(settlement.draw_number)
        if draw is None:
            continue
        by_tier = {payout.prize_tier: payout for payout in settlement.payouts}
        rows.append(
            MiniLotoEconomicResult(
                lottery=str(MINI_LOTO.code),
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                sales_amount_yen=_sales_amount_from_json(path),
                first_prize_winners=_tier_winners(by_tier, "1st"),
                first_prize_payout_yen=_tier_payout(by_tier, "1st"),
                second_prize_winners=_tier_winners(by_tier, "2nd"),
                second_prize_payout_yen=_tier_payout(by_tier, "2nd"),
                third_prize_winners=_tier_winners(by_tier, "3rd"),
                third_prize_payout_yen=_tier_payout(by_tier, "3rd"),
                fourth_prize_winners=_tier_winners(by_tier, "4th"),
                fourth_prize_payout_yen=_tier_payout(by_tier, "4th"),
                source=SOURCE_SETTLEMENT,
                source_url=f"file:{path.resolve()}",
                fetched_at=settlement.settled_at,
                source_quality="local_operational_settlement_copy",
            )
        )
    return tuple(rows)


def _tier_winners(by_tier: dict[str, Any], tier: str) -> int | None:
    payout = by_tier.get(tier)
    return None if payout is None else payout.winners_count


def _tier_payout(by_tier: dict[str, Any], tier: str) -> int | None:
    payout = by_tier.get(tier)
    return None if payout is None else payout.payout_yen


def _sales_amount_from_json(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sales_amount_yen", "sales_yen", "draw_sales_yen"):
        value = payload.get(key)
        if value is not None:
            return parse_yen(value)
    return None


def _first_int_attr(attrs: dict[str, str], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = parse_yen(attrs.get(name))
        if value is not None:
            return value
    return None


def _row_to_csv(row: MiniLotoEconomicResult) -> dict[str, str | int]:
    return {
        field: "" if getattr(row, field) is None else getattr(row, field)
        for field in ECONOMIC_FIELDNAMES
    }


def _scientific_values(row: MiniLotoEconomicResult) -> tuple[Any, ...]:
    return tuple(getattr(row, field) for field in ECONOMIC_FIELDNAMES[:12])


def _source_rank(source: str) -> int:
    return SOURCE_PRIORITY.get(source, 99)


def _compatible_missing_fill(
    preferred: MiniLotoEconomicResult,
    secondary: MiniLotoEconomicResult,
) -> bool:
    for field in ECONOMIC_FIELDNAMES[:12]:
        left = getattr(preferred, field)
        right = getattr(secondary, field)
        if left is not None and right is not None and left != right:
            return False
    return True


def _fill_missing(
    preferred: MiniLotoEconomicResult,
    secondary: MiniLotoEconomicResult,
) -> MiniLotoEconomicResult:
    values = {
        field: getattr(preferred, field)
        if getattr(preferred, field) is not None
        else getattr(secondary, field)
        for field in ECONOMIC_FIELDNAMES
    }
    return replace(preferred, **values)


def _usable_for_stage28(row: MiniLotoEconomicResult) -> bool:
    return (
        row.sales_amount_yen is not None
        and row.first_prize_winners is not None
        and row.first_prize_payout_yen is not None
    )


def _missing_ranges(
    canonical_draw_numbers: tuple[int, ...],
    covered_draw_numbers: set[int],
) -> tuple[tuple[int, int], ...]:
    missing = tuple(
        number for number in canonical_draw_numbers if number not in covered_draw_numbers
    )
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for number in missing:
        if start is None:
            start = previous = number
            continue
        if previous is not None and number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous if previous is not None else start))
        start = previous = number
    if start is not None:
        ranges.append((start, previous if previous is not None else start))
    return tuple(ranges)
