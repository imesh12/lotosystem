from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from backend.app.domain import MINI_LOTO
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import SourceDocument
from backend.app.research.persistence import research_result_json
from backend.app.research.result_sources import SMBC_SOURCE
from backend.app.research.stage28_economic_history import (
    SOURCE_SETTLEMENT,
    MiniLotoEconomicResult,
    build_mini_loto_economic_history,
    coverage_report,
    economic_rows_to_stage28_observations,
    load_economic_history_csv,
    merge_economic_results,
    parse_count,
    parse_smbc_mini_loto_economic_xml,
    parse_yen,
    write_economic_history_csv,
)
from backend.app.research.stage28_ticket_popularity import primary_association_test, score_universe


def _draw(number: int, numbers: tuple[int, ...] = (1, 4, 10, 20, 29)) -> HistoricalDraw:
    return HistoricalDraw(
        lottery=MINI_LOTO,
        draw_number=number,
        draw_date=date(2026, 1, 1),
        main_numbers=numbers,
        bonus_numbers=(31,),
    )


def _row(
    draw_number: int = 1401,
    *,
    sales: int | None = None,
    first_winners: int | None = 2,
    first_payout: int | None = 10_000_000,
    source: str = SOURCE_SETTLEMENT,
) -> MiniLotoEconomicResult:
    return MiniLotoEconomicResult(
        lottery=str(MINI_LOTO.code),
        draw_number=draw_number,
        draw_date="2026-01-01",
        sales_amount_yen=sales,
        first_prize_winners=first_winners,
        first_prize_payout_yen=first_payout,
        second_prize_winners=10,
        second_prize_payout_yen=100_000,
        third_prize_winners=100,
        third_prize_payout_yen=10_000,
        fourth_prize_winners=1000,
        fourth_prize_payout_yen=1000,
        source=source,
        source_url="test",
        fetched_at="2026-01-01T00:00:00+00:00",
        source_quality="test",
    )


def _settlement_payload(draw: HistoricalDraw, *, winners: int, payout: int) -> dict[str, object]:
    return {
        "schema_version": "stage12-paper-settlement-v1",
        "lottery": str(draw.lottery.code),
        "draw_number": draw.draw_number,
        "draw_date": draw.draw_date.isoformat(),
        "prediction_record_path": "prediction.json",
        "prediction_generated_at": "2026-01-01T00:00:00+00:00",
        "prediction_dataset_hash": "hash",
        "settled_at": "2026-01-02T00:00:00+00:00",
        "actual_main_numbers": draw.main_numbers,
        "actual_bonus_numbers": draw.bonus_numbers,
        "payouts": [
            {
                "lottery": str(draw.lottery.code),
                "draw_number": draw.draw_number,
                "prize_tier": "1st",
                "payout_yen": payout,
                "winners_count": winners,
                "source": "test",
                "source_url": None,
                "retrieved_at": None,
            }
        ],
        "tickets": [],
        "ticket_count": 0,
        "ticket_price_yen": draw.lottery.ticket_price_yen,
        "paper_total_cost_yen": 0,
        "paper_gross_winnings_yen": 0,
        "paper_net_yen": 0,
        "financial_status": "COMPLETE",
        "warnings": [],
    }


def test_parse_yen_and_winner_counts() -> None:
    assert parse_yen("1,234,500円") == 1_234_500
    assert parse_yen("￥2,000") == 2_000
    assert parse_count("12口") == 12
    assert parse_yen("") is None
    with pytest.raises(ResearchValidationError, match="invalid yen"):
        parse_yen("not-yen")


def test_parse_smbc_mini_loto_economic_xml_with_sales() -> None:
    draw = _draw(1401)
    document = SourceDocument(
        url="https://example.test/smbc.xml",
        text="",
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
        content_hash="hash",
    )
    xml = """
    <root>
      <data GAME_TYPE="04" KAIGOU="1401" TYUSEN_YMD="20260101"
        URIAGE_KINGAKU="123,456,000"
        TOUSEN_KUTI1="2" TOUSEN_KINGAKU1="10,000,000"
        TOUSEN_KUTI2="8" TOUSEN_KINGAKU2="150,000"
        TOUSEN_KUTI3="300" TOUSEN_KINGAKU3="10,000"
        TOUSEN_KUTI4="10000" TOUSEN_KINGAKU4="1,000" />
    </root>
    """

    rows = parse_smbc_mini_loto_economic_xml(xml, {1401: draw}, document)

    assert len(rows) == 1
    assert rows[0].sales_amount_yen == 123_456_000
    assert rows[0].first_prize_winners == 2
    assert rows[0].first_prize_payout_yen == 10_000_000


def test_malformed_smbc_xml_fails_clearly() -> None:
    document = SourceDocument(
        url="https://example.test/smbc.xml",
        text="",
        retrieved_at=datetime(2026, 1, 2, tzinfo=UTC),
        content_hash="hash",
    )

    with pytest.raises(ResearchValidationError, match="invalid SMBC economic XML"):
        parse_smbc_mini_loto_economic_xml("<root>", {}, document)


def test_duplicate_identical_rows_deduplicate() -> None:
    rows = merge_economic_results((_row(), _row()))

    assert rows == (_row(),)


def test_conflicting_rows_are_rejected() -> None:
    with pytest.raises(ResearchValidationError, match="conflicting Mini Loto economic record"):
        merge_economic_results((_row(first_winners=1), _row(first_winners=2)))


def test_source_precedence_fills_missing_values() -> None:
    settlement = _row(sales=None, source=SOURCE_SETTLEMENT)
    smbc = _row(sales=123_456_000, source=SMBC_SOURCE)

    rows = merge_economic_results((settlement, smbc))

    assert rows[0].source == SMBC_SOURCE
    assert rows[0].sales_amount_yen == 123_456_000
    assert rows[0].first_prize_winners == 2


def test_conflicting_higher_quality_source_fails() -> None:
    settlement = _row(first_winners=2, source=SOURCE_SETTLEMENT)
    smbc = _row(first_winners=3, source=SMBC_SOURCE)

    with pytest.raises(ResearchValidationError, match="conflicting Mini Loto economic record"):
        merge_economic_results((settlement, smbc))


def test_settlement_ingestion_preserves_missing_sales(tmp_path: Path) -> None:
    draw = _draw(1401)
    path = tmp_path / "settlements" / "MINI_LOTO" / "1401.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        research_result_json(_settlement_payload(draw, winners=2, payout=10_000_000)),
        encoding="utf-8",
    )

    rows = build_mini_loto_economic_history((draw,), settlement_root=tmp_path / "settlements")

    assert rows[0].first_prize_winners == 2
    assert rows[0].first_prize_payout_yen == 10_000_000
    assert rows[0].sales_amount_yen is None


def test_economic_csv_roundtrip_is_deterministic(tmp_path: Path) -> None:
    rows = (_row(draw_number=1402, sales=1000), _row(draw_number=1401, sales=2000))
    output = tmp_path / "economic.csv"

    first_path = write_economic_history_csv(merge_economic_results(rows), output)
    first = first_path.read_text(encoding="utf-8")
    loaded = load_economic_history_csv(output)
    write_economic_history_csv(loaded, output)

    assert first == output.read_text(encoding="utf-8")
    assert tuple(row.draw_number for row in loaded) == (1401, 1402)


def test_coverage_report_counts_usable_observations_and_missing_ranges() -> None:
    draws = (_draw(1401), _draw(1402), _draw(1403))
    rows = (_row(1401, sales=1000), _row(1403, sales=None))

    report = coverage_report(rows, draws)

    assert report.total_rows == 2
    assert report.rows_with_sales_amount == 1
    assert report.usable_stage28_observations == 1
    assert report.missing_ranges == ((1402, 1402),)


def test_stage28_observation_conversion_uses_sales_normalization() -> None:
    draw = _draw(1401)
    rows = (_row(1401, sales=2000, first_winners=2),)

    observations = economic_rows_to_stage28_observations(rows, {1401: draw}, score_universe())

    assert observations[0].estimated_tickets_sold == 10
    assert observations[0].normalized_winner_rate == 0.2
    assert observations[0].main_numbers == draw.main_numbers


def test_stage28_primary_endpoint_remains_inconclusive_without_sales() -> None:
    draw = _draw(1401)
    rows = (_row(1401, sales=None, first_winners=2),)
    observations = economic_rows_to_stage28_observations(rows, {1401: draw})

    result = primary_association_test(observations, seed=123456)

    assert result.usable_observations == 0
    assert result.classification == "INCONCLUSIVE"


def test_economic_stage_does_not_touch_stage27_production_or_settlements(tmp_path: Path) -> None:
    protected = (
        tmp_path / "prospective" / "stage27" / "sentinel.json",
        tmp_path / "predictions" / "MINI_LOTO" / "sentinel.json",
        tmp_path / "settlements" / "MINI_LOTO" / "sentinel.json",
    )
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"unchanged": true}', encoding="utf-8")
    before = {path: path.read_text(encoding="utf-8") for path in protected}

    rows = build_mini_loto_economic_history((), settlement_root=tmp_path / "empty")
    write_economic_history_csv(rows, tmp_path / "exports" / "economic.csv")

    assert {path: path.read_text(encoding="utf-8") for path in protected} == before
