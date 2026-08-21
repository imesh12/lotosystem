from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.browser_mizuho import (
    _deduplicate_browser_draws,
    _page_is_before_start_date,
    discover_mizuho_archive_ranges,
    discover_mizuho_recent_result_urls,
    generate_mizuho_archive_ranges,
)
from backend.app.research.collectors import CollectorInterface
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.dataset import calculate_dataset_hash
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import (
    _incremental_minimum_draw_number,
    find_missing_draw_numbers,
    merge_historical_draws,
    update_history,
    verify_history,
    write_canonical_history_csv,
)
from backend.app.research.mizuho import (
    MIZUHO_SOURCE,
    LocalMizuhoArchiveCollector,
    SourceDocument,
    filter_collected_draws,
    parse_mizuho_html_archive,
    parse_mizuho_rendered_rows,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RETRIEVED_AT = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)


class StaticCollector(CollectorInterface):
    def __init__(self, draws: tuple[HistoricalDraw, ...]) -> None:
        self.draws = draws

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        return tuple(draw for draw in self.draws if draw.lottery.code == lottery.code)


def _document(name: str, url: str = "https://www.mizuhobank.co.jp/test.html") -> SourceDocument:
    text = (FIXTURES_DIR / name).read_text(encoding="utf-8")
    return SourceDocument(url=url, text=text, retrieved_at=RETRIEVED_AT, content_hash="fixturehash")


def test_loto6_mizuho_archive_parsing() -> None:
    draws = parse_mizuho_html_archive(
        _document("mizuho_loto6_archive_sample.html").text,
        LOTO6,
        _document("mizuho_loto6_archive_sample.html"),
    )

    assert len(draws) == 2
    assert draws[0].draw_number == 484
    assert draws[0].draw_date == date(2010, 1, 4)
    assert draws[0].main_numbers == (1, 2, 3, 4, 5, 6)
    assert draws[0].bonus_numbers == (7,)
    assert draws[0].source == MIZUHO_SOURCE


def test_mini_loto_mizuho_archive_parsing() -> None:
    document = _document("mizuho_mini_loto_archive_sample.html")
    draws = parse_mizuho_html_archive(document.text, MINI_LOTO, document)

    assert len(draws) == 3
    assert draws[1].draw_number == 541
    assert draws[1].draw_date == date(2010, 1, 5)
    assert draws[1].main_numbers == (7, 8, 9, 10, 11)
    assert draws[1].bonus_numbers == (12,)


def test_loto6_rendered_table_extraction() -> None:
    document = SourceDocument(
        url="https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/detail.html",
        text="rendered",
        retrieved_at=RETRIEVED_AT,
        content_hash="renderedhash",
    )
    draws = parse_mizuho_rendered_rows(
        (
            ("回別", "抽せん日", "本数字", "ボーナス数字"),
            ("第484回", "2010年1月4日", "01 02 03 04 05 06", "07"),
            ("第485回", "2010年1月7日", "08 09 10 11 12 13", "14"),
        ),
        LOTO6,
        document,
    )

    assert [draw.draw_number for draw in draws] == [484, 485]
    assert draws[0].main_numbers == (1, 2, 3, 4, 5, 6)
    assert draws[0].bonus_numbers == (7,)


def test_loto6_recent_result_grouped_table_parsing() -> None:
    document = SourceDocument(
        url="https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=2026&month=7",
        text="rendered",
        retrieved_at=RETRIEVED_AT,
        content_hash="renderedhash",
    )
    draws = parse_mizuho_rendered_rows(
        (
            (
                "回別",
                "第2124回",
                "抽せん日",
                "2026年7月30日",
                "本数字",
                "06",
                "20",
                "29",
                "36",
                "37",
                "41",
                "ボーナス数字",
                "(19)",
                "1等",
                "該当なし",
            ),
        ),
        LOTO6,
        document,
    )

    assert len(draws) == 1
    assert draws[0].draw_number == 2124
    assert draws[0].draw_date == date(2026, 7, 30)
    assert draws[0].main_numbers == (6, 20, 29, 36, 37, 41)
    assert draws[0].bonus_numbers == (19,)


def test_mini_loto_rendered_table_extraction() -> None:
    document = SourceDocument(
        url="https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/detail.html",
        text="rendered",
        retrieved_at=RETRIEVED_AT,
        content_hash="renderedhash",
    )
    draws = parse_mizuho_rendered_rows(
        (
            ("第540回", "2009年12月29日", "01 02 03 04 05", "06"),
            ("第541回", "2010年1月5日", "07 08 09 10 11", "12"),
        ),
        MINI_LOTO,
        document,
    )

    filtered = filter_collected_draws(draws, MINI_LOTO, date(2010, 1, 1), date(2010, 1, 31), None)

    assert [draw.draw_number for draw in filtered] == [541]
    assert filtered[0].main_numbers == (7, 8, 9, 10, 11)
    assert filtered[0].bonus_numbers == (12,)


def test_mini_loto_recent_result_grouped_table_parsing() -> None:
    document = SourceDocument(
        url="https://www.mizuhobank.co.jp/takarakuji/check/loto/miniloto/index.html?year=2026&month=7",
        text="rendered",
        retrieved_at=RETRIEVED_AT,
        content_hash="renderedhash",
    )
    draws = parse_mizuho_rendered_rows(
        (
            (
                "回別",
                "第1397回",
                "抽せん日",
                "2026年7月28日",
                "本数字(　)はボーナス数字",
                "03",
                "12",
                "16",
                "20",
                "31",
                "(07)",
                "1等",
                "該当なし",
            ),
        ),
        MINI_LOTO,
        document,
    )

    assert len(draws) == 1
    assert draws[0].draw_number == 1397
    assert draws[0].draw_date == date(2026, 7, 28)
    assert draws[0].main_numbers == (3, 12, 16, 20, 31)
    assert draws[0].bonus_numbers == (7,)


def test_empty_rendered_table_extraction_returns_no_draws() -> None:
    document = SourceDocument(
        url="https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/detail.html",
        text="rendered",
        retrieved_at=RETRIEVED_AT,
        content_hash="renderedhash",
    )

    assert (
        parse_mizuho_rendered_rows(
            (("回別", "抽せん日", "本数字", "ボーナス数字"),), LOTO6, document
        )
        == ()
    )


def test_archive_range_discovery_deduplicates_and_filters_lottery_type() -> None:
    ranges = discover_mizuho_archive_ranges(
        (
            "/takarakuji/check/loto/backnumber/detail.html?fromto=481_500&type=loto6",
            "/takarakuji/check/loto/backnumber/detail.html?fromto=481_500&type=loto6",
            "/takarakuji/check/loto/backnumber/detail.html?fromto=541_560&type=miniloto",
        ),
        "loto6",
    )

    assert len(ranges) == 1
    assert ranges[0].start == 481
    assert ranges[0].end == 500
    assert ranges[0].url.startswith("https://www.mizuhobank.co.jp/")


def test_archive_range_discovery_supports_static_loto_links() -> None:
    loto6_ranges = discover_mizuho_archive_ranges(
        (
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto60481.html",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto0541.html",
        ),
        "loto6",
    )
    mini_ranges = discover_mizuho_archive_ranges(
        (
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto60481.html",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto0541.html",
        ),
        "miniloto",
    )

    assert [(archive_range.start, archive_range.end) for archive_range in loto6_ranges] == [
        (481, 500)
    ]
    assert [(archive_range.start, archive_range.end) for archive_range in mini_ranges] == [
        (541, 560)
    ]


def test_recent_result_url_discovery_supports_monthly_mizuho_links() -> None:
    urls = discover_mizuho_recent_result_urls(
        (
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/miniloto/index.html?year=2026&month=7",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=2025&month=12",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=2026&month=7",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html?year=2026&month=7",
            "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=bad&month=7",
        ),
        "loto6",
    )

    assert urls == (
        "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html",
        "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=2026&month=7",
        "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html?year=2025&month=12",
    )


def test_archive_range_fallback_generation_honors_incremental_minimum() -> None:
    ranges = generate_mizuho_archive_ranges("miniloto", minimum_draw_number=542, max_ranges=2)

    assert [(archive_range.start, archive_range.end) for archive_range in ranges] == [
        (541, 560),
        (561, 580),
    ]


def test_browser_page_before_start_date_detection() -> None:
    draws = (
        HistoricalDraw(MINI_LOTO, 541, date(2009, 12, 22), (1, 2, 3, 4, 5), (6,)),
        HistoricalDraw(MINI_LOTO, 542, date(2009, 12, 29), (7, 8, 9, 10, 11), (12,)),
    )

    assert _page_is_before_start_date(draws, date(2010, 1, 1)) is True
    assert _page_is_before_start_date(draws, date(2009, 12, 1)) is False


def test_mizuho_filter_applies_2010_cutoff_and_latest_completed_date() -> None:
    document = _document("mizuho_mini_loto_archive_sample.html")
    draws = parse_mizuho_html_archive(document.text, MINI_LOTO, document)

    filtered = filter_collected_draws(draws, MINI_LOTO, date(2010, 1, 1), date(2010, 1, 5), None)

    assert [draw.draw_number for draw in filtered] == [541]


def test_mizuho_filter_excludes_future_uncompleted_draws() -> None:
    draws = (
        HistoricalDraw(LOTO6, 2124, date(2026, 7, 30), (6, 20, 29, 36, 37, 41), (19,)),
        HistoricalDraw(LOTO6, 2125, date(2026, 8, 3), (1, 2, 3, 4, 5, 6), (7,)),
    )

    filtered = filter_collected_draws(draws, LOTO6, date(2010, 1, 1), date(2026, 7, 31), None)

    assert [draw.draw_number for draw in filtered] == [2124]


def test_malformed_mizuho_source_response_returns_no_draws() -> None:
    document = _document("mizuho_bad_archive_sample.html")

    assert parse_mizuho_html_archive(document.text, LOTO6, document) == ()


def test_chrome_saved_empty_mizuho_shell_returns_no_draws() -> None:
    document = _document("mizuho_chrome_saved_empty_shell.html")

    assert parse_mizuho_html_archive(document.text, LOTO6, document) == ()


def test_local_mizuho_collector_rejects_source_files_with_no_draw_records(tmp_path: Path) -> None:
    source_dir = tmp_path / "mizuho"
    source_dir.mkdir()
    (source_dir / "empty_shell.html").write_text(
        (FIXTURES_DIR / "mizuho_chrome_saved_empty_shell.html").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    collector = LocalMizuhoArchiveCollector(source_dir)

    with pytest.raises(ResearchValidationError, match="source files found.*no LOTO6 draw records"):
        collector.collect(LOTO6)


def test_mizuho_parser_rejects_duplicate_conflicting_rows() -> None:
    html = """
    <table>
      <tr><td>第484回</td><td>2010年1月4日</td><td>01</td><td>02</td><td>03</td>
      <td>04</td><td>05</td><td>06</td><td>07</td></tr>
      <tr><td>第484回</td><td>2010年1月4日</td><td>01</td><td>02</td><td>03</td>
      <td>04</td><td>05</td><td>08</td><td>07</td></tr>
    </table>
    """

    with pytest.raises(ResearchValidationError, match="conflicting Mizuho records"):
        parse_mizuho_html_archive(html, LOTO6, _document("mizuho_loto6_archive_sample.html"))


def test_merge_rejects_conflicting_historical_record() -> None:
    existing = (HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),)
    fetched = (HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 8), (7,)),)

    with pytest.raises(ResearchValidationError, match="conflicting historical record"):
        merge_historical_draws(existing, fetched)


def test_browser_draw_dedup_allows_overlapping_recent_and_bridge_sources() -> None:
    first = HistoricalDraw(LOTO6, 2022, date(2025, 8, 4), (1, 2, 3, 4, 5, 6), (7,))
    duplicate = HistoricalDraw(LOTO6, 2022, date(2025, 8, 4), (1, 2, 3, 4, 5, 6), (7,))

    assert _deduplicate_browser_draws([first, duplicate]) == (duplicate,)


def test_browser_draw_dedup_rejects_conflicting_overlap() -> None:
    first = HistoricalDraw(LOTO6, 2022, date(2025, 8, 4), (1, 2, 3, 4, 5, 6), (7,))
    conflict = HistoricalDraw(LOTO6, 2022, date(2025, 8, 4), (1, 2, 3, 4, 5, 8), (7,))

    with pytest.raises(ResearchValidationError, match="conflicting Mizuho browser records"):
        _deduplicate_browser_draws([first, conflict])


def test_canonical_write_and_dataset_hash_are_deterministic(tmp_path: Path) -> None:
    document = _document("mizuho_loto6_archive_sample.html")
    draws = parse_mizuho_html_archive(document.text, LOTO6, document)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    write_canonical_history_csv(draws, first_path)
    write_canonical_history_csv(tuple(reversed(draws)), second_path)

    first = load_draws_csv(first_path, LOTO6)
    second = load_draws_csv(second_path, LOTO6)
    assert calculate_dataset_hash(first) == calculate_dataset_hash(second)
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_update_history_is_idempotent_for_unchanged_source(tmp_path: Path) -> None:
    document = _document("mizuho_loto6_archive_sample.html")
    draws = parse_mizuho_html_archive(document.text, LOTO6, document)
    output_path = tmp_path / "loto6_history.csv"

    first = update_history(output_path, LOTO6, StaticCollector(draws))
    second = update_history(output_path, LOTO6, StaticCollector(draws))

    assert first.written_count == 2
    assert second.written_count == 2
    assert second.appended_count == 0
    assert second.unchanged_count == 2
    assert first.verification.dataset_hash == second.verification.dataset_hash


def test_update_history_already_current_no_op_keeps_existing_file(tmp_path: Path) -> None:
    draws = (
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 485, date(2010, 1, 7), (8, 9, 10, 11, 12, 13), (14,)),
    )
    output_path = tmp_path / "loto6_history.csv"
    write_canonical_history_csv(draws, output_path)

    result = update_history(output_path, LOTO6, StaticCollector((draws[-1],)))

    assert result.appended_count == 0
    assert result.unchanged_count == 1
    assert result.written_count == 2


def test_update_history_appends_incremental_new_draw(tmp_path: Path) -> None:
    existing = (HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),)
    fetched = existing + (
        HistoricalDraw(LOTO6, 485, date(2010, 1, 7), (8, 9, 10, 11, 12, 13), (14,)),
    )
    output_path = tmp_path / "loto6_history.csv"
    write_canonical_history_csv(existing, output_path)

    result = update_history(output_path, LOTO6, StaticCollector(fetched))

    assert result.existing_count == 1
    assert result.appended_count == 1
    assert result.written_count == 2


def test_update_history_can_repair_gap_older_than_latest_draw(tmp_path: Path) -> None:
    existing = (
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 486, date(2010, 1, 11), (8, 9, 10, 11, 12, 13), (14,)),
    )
    fetched = (HistoricalDraw(LOTO6, 485, date(2010, 1, 7), (15, 16, 17, 18, 19, 20), (21,)),)
    output_path = tmp_path / "loto6_history.csv"
    write_canonical_history_csv(existing, output_path)

    result = update_history(output_path, LOTO6, StaticCollector(fetched))
    draws = load_draws_csv(output_path, LOTO6)

    assert result.appended_count == 1
    assert [draw.draw_number for draw in draws] == [484, 485, 486]
    assert result.verification.missing_draw_numbers == ()


def test_update_history_uses_offline_mocked_collector(tmp_path: Path) -> None:
    draws = (
        HistoricalDraw(MINI_LOTO, 541, date(2010, 1, 5), (7, 8, 9, 10, 11), (12,)),
        HistoricalDraw(MINI_LOTO, 542, date(2010, 1, 12), (13, 14, 15, 16, 17), (18,)),
    )

    result = update_history(tmp_path / "mini_loto_history.csv", MINI_LOTO, StaticCollector(draws))

    assert result.verification.lottery == "MINI_LOTO"
    assert result.verification.validation_errors == ()
    assert result.verification.first_draw_number == 541


def test_local_mizuho_archive_collector_parses_saved_official_files() -> None:
    collector = LocalMizuhoArchiveCollector(
        FIXTURES_DIR,
        start_date=date(2010, 1, 1),
        end_date=date(2010, 1, 12),
    )

    loto6_draws = collector.collect(LOTO6)
    mini_draws = collector.collect(MINI_LOTO)

    assert [draw.draw_number for draw in loto6_draws] == [484, 485]
    assert [draw.draw_number for draw in mini_draws] == [541, 542]
    assert all(draw.source_url.startswith("file:") for draw in loto6_draws + mini_draws)


def test_local_mizuho_archive_update_writes_canonical_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "mini_loto_history.csv"
    collector = LocalMizuhoArchiveCollector(
        FIXTURES_DIR,
        start_date=date(2010, 1, 1),
        end_date=date(2010, 1, 12),
    )

    result = update_history(output_path, MINI_LOTO, collector)
    draws = load_draws_csv(output_path, MINI_LOTO)

    assert result.written_count == 2
    assert draws[-1].draw_number == 542
    assert result.verification.dataset_hash == calculate_dataset_hash(draws)


def test_verify_history_reports_duplicates() -> None:
    draws = (
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
    )

    verification = verify_history(draws, LOTO6)

    assert verification.duplicate_draw_numbers == (484,)
    assert verification.duplicate_records
    assert verification.validation_errors


def test_verify_history_reports_missing_draw_number_gap() -> None:
    draws = (
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 486, date(2010, 1, 11), (8, 9, 10, 11, 12, 13), (14,)),
    )

    verification = verify_history(draws, LOTO6)

    assert find_missing_draw_numbers(draws) == (485,)
    assert verification.missing_draw_numbers == (485,)


def test_incremental_start_targets_existing_gap_before_latest_draw() -> None:
    draws = (
        HistoricalDraw(LOTO6, 484, date(2010, 1, 4), (1, 2, 3, 4, 5, 6), (7,)),
        HistoricalDraw(LOTO6, 486, date(2010, 1, 11), (8, 9, 10, 11, 12, 13), (14,)),
        HistoricalDraw(LOTO6, 487, date(2010, 1, 14), (15, 16, 17, 18, 19, 20), (21,)),
    )

    assert _incremental_minimum_draw_number(draws) == 484


def test_cli_validate_data_rejects_empty_history_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "lottery,draw_number,draw_date,n1,n2,n3,n4,n5,n6,bonus\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "LOTO6",
            "--data",
            str(csv_path),
            "validate-data",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["status"] == "error"
    assert "no LOTO6 draw records" in payload["error"]
