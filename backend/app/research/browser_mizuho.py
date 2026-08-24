from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import parse_qs, urljoin, urlparse

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.collectors import CollectorInterface
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import (
    DEFAULT_HISTORY_START,
    MIZUHO_DETAIL_URL,
    SourceDocument,
    filter_collected_draws,
    parse_mizuho_rendered_rows,
)

MIZUHO_BACKNUMBER_INDEX_URL = (
    "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/index.html"
)
MIZUHO_CURRENT_RESULT_URLS = {
    "loto6": "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto6/index.html",
    "miniloto": "https://www.mizuhobank.co.jp/takarakuji/check/loto/miniloto/index.html",
}
INCREMENTAL_BRIDGE_RANGE_COUNT = 12
PAGE_STATE_ACCESS_DENIED = "ACCESS_DENIED"
PAGE_STATE_EMPTY_OR_LOADING = "EMPTY_OR_LOADING"
PAGE_STATE_RENDERED = "RENDERED"


@dataclass(frozen=True, slots=True)
class MizuhoArchiveRange:
    start: int
    end: int
    url: str


class BrowserMizuhoCollector(CollectorInterface):
    """Collect JS-rendered Mizuho archive tables with a local Playwright browser."""

    def __init__(
        self,
        *,
        start_date: date = DEFAULT_HISTORY_START,
        end_date: date | None = None,
        minimum_draw_number: int | None = None,
        headed: bool = False,
        max_ranges: int = 250,
        timeout_ms: int = 30_000,
        row_timeout_ms: int = 7_000,
        index_url: str = MIZUHO_BACKNUMBER_INDEX_URL,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.minimum_draw_number = minimum_draw_number
        self.headed = headed
        self.max_ranges = max_ranges
        self.timeout_ms = timeout_ms
        self.row_timeout_ms = row_timeout_ms
        self.index_url = index_url

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ResearchValidationError(
                "Playwright is required for browser bootstrap; install project dependencies and "
                "run `python -m playwright install chromium`"
            ) from exc

        lottery_type = _lottery_type(lottery)
        draws: list[HistoricalDraw] = []

        try:
            with sync_playwright() as playwright:
                browser = _launch_browser(playwright, headed=self.headed)
                page = browser.new_page(locale="ja-JP", timezone_id="Asia/Tokyo")
                page.set_default_timeout(self.timeout_ms)
                recent_urls = self._discover_recent_urls(page, lottery_type)
                ranges = self._discover_ranges(page, lottery_type)

                for url in recent_urls:
                    page.goto(url, wait_until="domcontentloaded")
                    _raise_for_source_failure(page, url)
                    if not _wait_for_rendered_rows(page, timeout_ms=self.row_timeout_ms):
                        continue
                    document = _document_from_page(page, url)
                    rendered_rows = _extract_rendered_rows(page)
                    draws.extend(parse_mizuho_rendered_rows(rendered_rows, lottery, document))

                for archive_range in ranges:
                    page.goto(archive_range.url, wait_until="domcontentloaded")
                    _raise_for_source_failure(page, archive_range.url)
                    if not _wait_for_rendered_rows(page, timeout_ms=self.row_timeout_ms):
                        if (
                            self.minimum_draw_number is not None
                            and archive_range.start >= self.minimum_draw_number
                        ):
                            break
                        continue
                    document = _document_from_page(page, archive_range.url)
                    rendered_rows = _extract_rendered_rows(page)
                    page_draws = parse_mizuho_rendered_rows(rendered_rows, lottery, document)
                    if _page_is_before_start_date(page_draws, self.start_date):
                        break
                    draws.extend(page_draws)
                browser.close()
        except (PlaywrightError, PlaywrightTimeoutError) as exc:
            raise ResearchValidationError(f"Mizuho browser bootstrap failed: {exc}") from exc

        if not draws and self.minimum_draw_number is not None:
            return ()

        if not draws:
            raise ResearchValidationError(
                f"browser bootstrap found no rendered {lottery.code} draw records"
            )

        return filter_collected_draws(
            _deduplicate_browser_draws(draws),
            lottery,
            self.start_date,
            self.end_date,
            self.minimum_draw_number,
        )

    def _discover_ranges(self, page: object, lottery_type: str) -> tuple[MizuhoArchiveRange, ...]:
        page.goto(self.index_url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        except Exception:
            pass
        _raise_for_source_failure(page, self.index_url)
        hrefs = page.locator("a[href]").evaluate_all(
            "links => links.map(link => link.href).filter(Boolean)"
        )
        ranges = discover_mizuho_archive_ranges(hrefs, lottery_type)
        if not ranges:
            ranges = generate_mizuho_archive_ranges(
                lottery_type,
                minimum_draw_number=self.minimum_draw_number,
                max_ranges=self.max_ranges,
            )
        elif self.minimum_draw_number is not None:
            ranges = _merge_archive_ranges(
                ranges,
                generate_mizuho_archive_ranges(
                    lottery_type,
                    minimum_draw_number=self.minimum_draw_number,
                    max_ranges=min(INCREMENTAL_BRIDGE_RANGE_COUNT, self.max_ranges),
                ),
            )
        ranges = tuple(
            archive_range
            for archive_range in ranges
            if self.minimum_draw_number is None or archive_range.end >= self.minimum_draw_number
        )
        if self.minimum_draw_number is None:
            ranges = tuple(reversed(ranges))
        return ranges[: self.max_ranges]

    def _discover_recent_urls(self, page: object, lottery_type: str) -> tuple[str, ...]:
        page.goto(self.index_url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        except Exception:
            pass
        _raise_for_source_failure(page, self.index_url)
        hrefs = page.locator("a[href]").evaluate_all(
            "links => links.map(link => link.href).filter(Boolean)"
        )
        return discover_mizuho_recent_result_urls(hrefs, lottery_type)


def discover_mizuho_archive_ranges(
    hrefs: list[str] | tuple[str, ...],
    lottery_type: str,
    *,
    base_url: str = MIZUHO_BACKNUMBER_INDEX_URL,
) -> tuple[MizuhoArchiveRange, ...]:
    ranges: dict[tuple[int, int], MizuhoArchiveRange] = {}
    for href in hrefs:
        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)
        query = parse_qs(parsed.query)
        if query.get("type", ("",))[0] != lottery_type:
            static_range = _static_archive_range(absolute_url, lottery_type)
            if static_range is None:
                continue
            start, end = static_range
        else:
            fromto = query.get("fromto", ("",))[0]
            match = re.fullmatch(r"(\d+)_(\d+)", fromto)
            if not match:
                continue
            start = int(match.group(1))
            end = int(match.group(2))
        ranges[(start, end)] = MizuhoArchiveRange(start=start, end=end, url=absolute_url)
    return tuple(ranges[key] for key in sorted(ranges))


def _merge_archive_ranges(
    first: tuple[MizuhoArchiveRange, ...],
    second: tuple[MizuhoArchiveRange, ...],
) -> tuple[MizuhoArchiveRange, ...]:
    ranges = {(archive_range.start, archive_range.end): archive_range for archive_range in first}
    for archive_range in second:
        ranges.setdefault((archive_range.start, archive_range.end), archive_range)
    return tuple(ranges[key] for key in sorted(ranges))


def discover_mizuho_recent_result_urls(
    hrefs: list[str] | tuple[str, ...],
    lottery_type: str,
    *,
    base_url: str = MIZUHO_BACKNUMBER_INDEX_URL,
) -> tuple[str, ...]:
    current_url = MIZUHO_CURRENT_RESULT_URLS.get(lottery_type)
    if current_url is None:
        return ()

    month_urls: dict[tuple[int, int], str] = {}
    expected_path = urlparse(current_url).path
    for href in hrefs:
        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)
        if parsed.path != expected_path:
            continue
        query = parse_qs(parsed.query)
        if "year" not in query or "month" not in query:
            continue
        try:
            year = int(query["year"][0])
            month = int(query["month"][0])
        except (TypeError, ValueError):
            continue
        if 1 <= month <= 12:
            month_urls[(year, month)] = absolute_url

    ordered_month_urls = tuple(month_urls[key] for key in sorted(month_urls, reverse=True))
    return (current_url, *ordered_month_urls)


def generate_mizuho_archive_ranges(
    lottery_type: str,
    *,
    minimum_draw_number: int | None = None,
    max_ranges: int = 250,
) -> tuple[MizuhoArchiveRange, ...]:
    first_start = 1
    if minimum_draw_number is not None:
        first_start = ((max(1, minimum_draw_number) - 1) // 20) * 20 + 1

    ranges: list[MizuhoArchiveRange] = []
    for index in range(max_ranges):
        start = first_start + index * 20
        end = start + 19
        ranges.append(
            MizuhoArchiveRange(
                start=start,
                end=end,
                url=MIZUHO_DETAIL_URL.format(
                    start=start,
                    end=end,
                    lottery_type=lottery_type,
                ),
            )
        )
    return tuple(ranges)


def _launch_browser(playwright: object, *, headed: bool) -> object:
    for channel in ("chrome", "msedge", None):
        try:
            kwargs = {"headless": not headed}
            if channel is not None:
                kwargs["channel"] = channel
            return playwright.chromium.launch(**kwargs)
        except Exception:
            continue
    raise ResearchValidationError(
        "could not launch Chrome, Edge, or Playwright Chromium; run "
        "`python -m playwright install chromium`"
    )


def _static_archive_range(url: str, lottery_type: str) -> tuple[int, int] | None:
    path = urlparse(url).path
    if lottery_type == "loto6":
        match = re.search(r"/loto6(\d{4})\.html$", path)
    elif lottery_type == "miniloto":
        match = re.search(r"/loto(\d{4})\.html$", path)
    else:
        match = None
    if match is None:
        return None
    start = int(match.group(1))
    return start, start + 19


def _wait_for_rendered_rows(page: object, *, timeout_ms: int) -> bool:
    try:
        page.wait_for_function(
            """() => Array.from(document.querySelectorAll(
            '.js-lottery-backnumber-list table.pc-only tr, table tr'
        )).some(row => /第\\s*\\d+\\s*回/.test(row.innerText))""",
            timeout=timeout_ms,
        )
    except Exception:
        return False
    return True


def classify_mizuho_page_state(
    *,
    title: str,
    body_text: str,
    table_count: int,
    row_count: int,
) -> str:
    normalized_title = title.casefold()
    normalized_body = body_text.casefold()
    if (
        "access denied" in normalized_title
        or "access denied" in normalized_body
        or "permission to access" in normalized_body
        or "errors.edgesuite.net" in normalized_body
    ):
        return PAGE_STATE_ACCESS_DENIED
    if table_count == 0 and row_count == 0:
        return PAGE_STATE_EMPTY_OR_LOADING
    return PAGE_STATE_RENDERED


def _raise_for_source_failure(page: object, url: str) -> None:
    title = page.title()
    try:
        body_text = page.locator("body").inner_text(timeout=2_000)
    except Exception:
        body_text = ""
    table_count = page.locator("table").count()
    row_count = page.locator("table tr").count()
    state = classify_mizuho_page_state(
        title=title,
        body_text=body_text,
        table_count=table_count,
        row_count=row_count,
    )
    if state == PAGE_STATE_ACCESS_DENIED:
        raise ResearchValidationError(f"Mizuho source failure at {url}: access denied")
    if state == PAGE_STATE_EMPTY_OR_LOADING and _is_result_or_detail_url(url):
        raise ResearchValidationError(
            f"Mizuho source failure at {url}: rendered page contained no result table"
        )


def _extract_rendered_rows(page: object) -> tuple[tuple[str, ...], ...]:
    rows = page.locator(".js-lottery-backnumber-list table.pc-only tr, table tr").evaluate_all(
        """rows => rows.map(row => Array.from(row.querySelectorAll('th,td'))
            .map(cell => cell.innerText.trim())
            .filter(Boolean))"""
    )
    draw_tables = page.locator("table").evaluate_all(
        """tables => {
            const drawPattern = /第\\s*\\d+\\s*回/;
            const datePattern = /\\d{4}\\s*年\\s*\\d{1,2}\\s*月\\s*\\d{1,2}\\s*日/;
            return tables.map(table => Array.from(table.querySelectorAll('th,td'))
                .map(cell => cell.innerText.trim())
                .filter(Boolean))
                .filter(cells => cells.filter(cell => drawPattern.test(cell)).length === 1
                    && cells.some(cell => datePattern.test(cell)));
        }"""
    )
    combined = [*rows, *draw_tables]
    return tuple(tuple(str(cell) for cell in row) for row in combined if row)


def _is_result_or_detail_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.path.endswith("/detail.html")
        or parsed.path.endswith("/loto6/index.html")
        or parsed.path.endswith("/miniloto/index.html")
    )


def _document_from_page(page: object, url: str) -> SourceDocument:
    text = page.content()
    return SourceDocument(
        url=url,
        text=text,
        retrieved_at=datetime.now(UTC),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _lottery_type(lottery: LotteryDefinition) -> str:
    if lottery.code == LOTO6.code:
        return "loto6"
    if lottery.code == MINI_LOTO.code:
        return "miniloto"
    raise ResearchValidationError(f"unsupported Mizuho browser lottery: {lottery.code}")


def _deduplicate_browser_draws(draws: list[HistoricalDraw]) -> tuple[HistoricalDraw, ...]:
    seen: dict[tuple[str, int], HistoricalDraw] = {}
    for draw in draws:
        identity = (str(draw.lottery.code), draw.draw_number)
        existing = seen.get(identity)
        if existing is not None and existing.canonical_identity != draw.canonical_identity:
            raise ResearchValidationError(
                f"conflicting Mizuho browser records for {identity[0]} #{identity[1]}"
            )
        seen[identity] = draw
    return tuple(sorted(seen.values(), key=lambda draw: (draw.draw_date, draw.draw_number)))


def _page_is_before_start_date(
    draws: tuple[HistoricalDraw, ...],
    start_date: date,
) -> bool:
    return bool(draws) and max(draw.draw_date for draw in draws) < start_date
