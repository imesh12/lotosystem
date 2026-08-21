from __future__ import annotations

import csv
import hashlib
import html
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from http.cookiejar import CookieJar
from pathlib import Path

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.collectors import CollectorInterface
from backend.app.research.data import HistoricalDraw, validate_draw_sequence
from backend.app.research.exceptions import ResearchValidationError

MIZUHO_SOURCE = "mizuho_bank"
DEFAULT_HISTORY_START = date(2010, 1, 1)
MIZUHO_LOTO6_CSV_URL = "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto6/csv/loto6.csv"
MIZUHO_LOTO6_HTML_URL = (
    "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto6{start:04d}.html"
)
MIZUHO_MINI_LOTO_HTML_URL = (
    "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/loto{start:04d}.html"
)
MIZUHO_DETAIL_URL = (
    "https://www.mizuhobank.co.jp/takarakuji/check/loto/backnumber/"
    "detail.html?fromto={start}_{end}&type={lottery_type}"
)
MIZUHO_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,text/csv;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Connection": "close",
}

_FULLWIDTH_TRANSLATION = str.maketrans("０１２３４５６７８９", "0123456789")


@dataclass(frozen=True, slots=True)
class SourceDocument:
    url: str
    text: str
    retrieved_at: datetime
    content_hash: str


class MizuhoFetchError(ResearchValidationError):
    """Raised when an authoritative Mizuho response cannot be fetched."""


class MizuhoHistoricalCollector(CollectorInterface):
    """Collect authoritative historical draws from Mizuho Bank lottery archives."""

    def __init__(
        self,
        *,
        start_date: date = DEFAULT_HISTORY_START,
        end_date: date | None = None,
        minimum_draw_number: int | None = None,
        timeout_seconds: float = 20.0,
        max_html_pages: int = 250,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.minimum_draw_number = minimum_draw_number
        self.timeout_seconds = timeout_seconds
        self.max_html_pages = max_html_pages

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        if lottery.code == LOTO6.code:
            draws = self._collect_loto6()
        elif lottery.code == MINI_LOTO.code:
            draws = self._collect_mini_loto()
        else:
            raise ResearchValidationError(f"unsupported Mizuho lottery: {lottery.code}")
        return filter_collected_draws(
            draws,
            lottery,
            self.start_date,
            self.end_date,
            self.minimum_draw_number,
        )

    def _collect_loto6(self) -> tuple[HistoricalDraw, ...]:
        csv_error: MizuhoFetchError | None = None
        try:
            document = fetch_mizuho_document(MIZUHO_LOTO6_CSV_URL, self.timeout_seconds)
            draws = parse_mizuho_csv_archive(document.text, LOTO6, document)
            if draws:
                return draws
        except MizuhoFetchError as exc:
            csv_error = exc

        try:
            return self._collect_html_pages(
                LOTO6,
                (MIZUHO_DETAIL_URL, MIZUHO_LOTO6_HTML_URL),
                lottery_type="loto6",
            )
        except ResearchValidationError as exc:
            if csv_error is not None:
                raise ResearchValidationError(f"{exc}; CSV archive error: {csv_error}") from exc
            raise

    def _collect_mini_loto(self) -> tuple[HistoricalDraw, ...]:
        return self._collect_html_pages(
            MINI_LOTO,
            (MIZUHO_DETAIL_URL, MIZUHO_MINI_LOTO_HTML_URL),
            lottery_type="miniloto",
        )

    def _collect_html_pages(
        self,
        lottery: LotteryDefinition,
        url_templates: tuple[str, ...],
        *,
        lottery_type: str,
    ) -> tuple[HistoricalDraw, ...]:
        draws: list[HistoricalDraw] = []
        start = _first_page_start(self.minimum_draw_number)
        empty_pages = 0
        fetch_errors: list[str] = []

        for _ in range(self.max_html_pages):
            page_draws: tuple[HistoricalDraw, ...] = ()
            for url_template in url_templates:
                url = url_template.format(
                    start=start,
                    end=start + 19,
                    lottery_type=lottery_type,
                )
                try:
                    document = fetch_mizuho_document(url, self.timeout_seconds)
                except MizuhoFetchError as exc:
                    fetch_errors.append(str(exc))
                    continue
                page_draws = parse_mizuho_html_archive(document.text, lottery, document)
                page_draws = tuple(
                    draw
                    for draw in page_draws
                    if _draw_number_allowed(draw, self.minimum_draw_number)
                )
                if page_draws:
                    break
            if page_draws:
                empty_pages = 0
                draws.extend(page_draws)
            else:
                empty_pages += 1
                if empty_pages >= 5:
                    break
            start += 20

        if not draws:
            detail = f"; first fetch error: {fetch_errors[0]}" if fetch_errors else ""
            raise ResearchValidationError(
                f"no Mizuho archive draws found for {lottery.code}{detail}"
            )
        return tuple(draws)


class LocalMizuhoArchiveCollector(CollectorInterface):
    """Parse locally saved official Mizuho archive files."""

    def __init__(
        self,
        source_dir: str | Path,
        *,
        start_date: date = DEFAULT_HISTORY_START,
        end_date: date | None = None,
        minimum_draw_number: int | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.minimum_draw_number = minimum_draw_number

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        if not self.source_dir.exists():
            raise ResearchValidationError(
                f"Mizuho source directory does not exist: {self.source_dir}"
            )
        documents = tuple(_load_local_mizuho_documents(self.source_dir))
        if not documents:
            raise ResearchValidationError(f"no Mizuho source files found in {self.source_dir}")

        draws: list[HistoricalDraw] = []
        for document in documents:
            if not _document_may_contain_lottery(document.text, lottery):
                continue
            if document.url.lower().endswith(".csv"):
                draws.extend(parse_mizuho_csv_archive(document.text, lottery, document))
            else:
                draws.extend(parse_mizuho_html_archive(document.text, lottery, document))

        if not draws:
            raise ResearchValidationError(
                f"source files found in {self.source_dir}, but no {lottery.code} draw records "
                "were parsed; save the populated Mizuho result page or source data file"
            )

        filtered = filter_collected_draws(
            _deduplicate_source_rows(draws),
            lottery,
            self.start_date,
            self.end_date,
            self.minimum_draw_number,
        )
        if not filtered:
            raise ResearchValidationError(
                f"parsed {len(draws)} {lottery.code} draw records, but none matched "
                f"date range {self.start_date.isoformat()} through {self.end_date.isoformat()}"
            )
        return filtered


def fetch_mizuho_document(url: str, timeout_seconds: float = 20.0) -> SourceDocument:
    request = urllib.request.Request(
        url,
        headers=MIZUHO_REQUEST_HEADERS,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = response.read()
            content_type = response.headers.get_content_charset()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise MizuhoFetchError(f"failed to fetch Mizuho source {url}: {exc}") from exc

    text = _decode_source_bytes(raw, content_type)
    return SourceDocument(
        url=url,
        text=text,
        retrieved_at=datetime.now(UTC),
        content_hash=hashlib.sha256(raw).hexdigest(),
    )


def _load_local_mizuho_documents(source_dir: Path) -> Iterable[SourceDocument]:
    supported_suffixes = {".html", ".htm", ".csv", ".txt"}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        raw = path.read_bytes()
        yield SourceDocument(
            url=f"file:{path.resolve()}",
            text=_decode_source_bytes(raw, None),
            retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
            content_hash=hashlib.sha256(raw).hexdigest(),
        )


def _document_may_contain_lottery(text: str, lottery: LotteryDefinition) -> bool:
    normalized = _clean_text(text).lower()
    has_loto6 = "ロト6" in normalized or "loto6" in normalized
    has_mini = "ミニロト" in normalized or "mini loto" in normalized or "miniloto" in normalized
    if lottery.code == LOTO6.code and has_mini and not has_loto6:
        return False
    if lottery.code == MINI_LOTO.code and has_loto6 and not has_mini:
        return False
    return True


def parse_mizuho_html_archive(
    html_text: str,
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for fields in _extract_html_table_rows(html_text):
        draw = _parse_draw_fields(fields, lottery, document)
        if draw is not None:
            rows.append(draw)

    if not rows:
        for line in _plain_text_lines(html_text):
            draw = _parse_draw_fields(_line_to_fields(line), lottery, document)
            if draw is not None:
                rows.append(draw)

    return _deduplicate_source_rows(rows)


def parse_mizuho_rendered_rows(
    rendered_rows: Iterable[Iterable[str]],
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for fields in rendered_rows:
        draw = _parse_draw_fields(fields, lottery, document)
        if draw is not None:
            rows.append(draw)
    return _deduplicate_source_rows(rows)


def parse_mizuho_csv_archive(
    csv_text: str,
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for fields in csv.reader(csv_text.splitlines()):
        draw = _parse_draw_fields(fields, lottery, document)
        if draw is not None:
            rows.append(draw)
    return _deduplicate_source_rows(rows)


def _decode_source_bytes(raw: bytes, charset: str | None) -> str:
    encodings = tuple(dict.fromkeys((charset, "cp932", "shift_jis", "utf-8", "euc_jp")))
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_html_table_rows(html_text: str) -> Iterable[tuple[str, ...]]:
    for row_html in re.findall(
        r"<tr\b[^>]*>(.*?)</tr>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        fields = tuple(
            _clean_text(cell)
            for cell in re.findall(
                r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
                row_html,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if fields:
            yield fields


def _plain_text_lines(html_text: str) -> tuple[str, ...]:
    without_tags = re.sub(r"<[^>]+>", "\n", html_text)
    plain = _clean_text(without_tags)
    return tuple(line.strip() for line in plain.splitlines() if line.strip())


def _line_to_fields(line: str) -> tuple[str, ...]:
    if "|" in line or "," in line or "\t" in line:
        return tuple(field.strip() for field in re.split(r"[|,\t]+", line) if field.strip())
    return (line,)


def _clean_text(value: str) -> str:
    cleaned = html.unescape(value)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.translate(_FULLWIDTH_TRANSLATION)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = cleaned.replace("\xa0", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_draw_fields(
    raw_fields: Iterable[str],
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> HistoricalDraw | None:
    fields = tuple(_clean_text(field) for field in raw_fields if _clean_text(field))
    if not fields:
        return None

    draw_number = _find_draw_number(fields)
    draw_date = _find_draw_date(fields)
    if draw_number is None or draw_date is None:
        return None

    date_index = _find_draw_date_index(fields)
    number_fields = fields[date_index + 1 :] if date_index is not None else fields
    source_numbers = _extract_small_numbers(number_fields, lottery)
    needed = lottery.numbers_per_ticket + lottery.bonus_numbers
    if len(source_numbers) < needed:
        return None

    main_numbers = tuple(source_numbers[: lottery.numbers_per_ticket])
    bonus_numbers = tuple(source_numbers[lottery.numbers_per_ticket : needed])
    return HistoricalDraw(
        lottery=lottery,
        draw_number=draw_number,
        draw_date=draw_date,
        main_numbers=main_numbers,
        bonus_numbers=bonus_numbers,
        source=MIZUHO_SOURCE,
        source_url=document.url,
        retrieved_at=document.retrieved_at,
        content_hash=document.content_hash,
    )


def _find_draw_number(fields: tuple[str, ...]) -> int | None:
    for field in fields:
        match = re.search(r"第\s*(\d+)\s*回", field)
        if match:
            return int(match.group(1))
    return None


def _find_draw_date(fields: tuple[str, ...]) -> date | None:
    for field in fields:
        match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", field)
        if match:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        iso_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", field)
        if iso_match:
            return date(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
    return None


def _find_draw_date_index(fields: tuple[str, ...]) -> int | None:
    for index, field in enumerate(fields):
        if re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日", field) or re.search(
            r"\d{4}-\d{1,2}-\d{1,2}",
            field,
        ):
            return index
    return None


def _extract_small_numbers(fields: tuple[str, ...], lottery: LotteryDefinition) -> list[int]:
    numbers: list[int] = []
    for field in fields:
        for raw_number in re.findall(r"\b\d{1,2}\b", field):
            number = int(raw_number)
            if lottery.number_min <= number <= lottery.number_max:
                numbers.append(number)
            if len(numbers) >= lottery.numbers_per_ticket + lottery.bonus_numbers:
                return numbers
    return numbers


def _deduplicate_source_rows(rows: list[HistoricalDraw]) -> tuple[HistoricalDraw, ...]:
    seen: dict[tuple[str, int], HistoricalDraw] = {}
    for draw in rows:
        identity = (str(draw.lottery.code), draw.draw_number)
        existing = seen.get(identity)
        if existing is not None and existing.canonical_identity != draw.canonical_identity:
            raise ResearchValidationError(
                f"conflicting Mizuho records for {identity[0]} #{identity[1]}"
            )
        seen[identity] = draw
    return tuple(sorted(seen.values(), key=lambda draw: (draw.draw_date, draw.draw_number)))


def filter_collected_draws(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    start_date: date,
    end_date: date,
    minimum_draw_number: int | None,
) -> tuple[HistoricalDraw, ...]:
    filtered = tuple(
        draw
        for draw in draws
        if draw.lottery.code == lottery.code
        and start_date <= draw.draw_date <= end_date
        and _draw_number_allowed(draw, minimum_draw_number)
    )
    return validate_draw_sequence(filtered)


def _draw_number_allowed(draw: HistoricalDraw, minimum_draw_number: int | None) -> bool:
    return minimum_draw_number is None or draw.draw_number >= minimum_draw_number


def _first_page_start(minimum_draw_number: int | None) -> int:
    if minimum_draw_number is None:
        return 1
    return ((max(minimum_draw_number, 1) - 1) // 20) * 20 + 1
