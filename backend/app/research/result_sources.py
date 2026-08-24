from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.collectors import CollectorInterface
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import (
    DEFAULT_HISTORY_START,
    SourceDocument,
    fetch_mizuho_document,
    filter_collected_draws,
)

SMBC_SOURCE = "smbc_public_result"
SMBC_LOTO6_XML_URL = "https://www.smbc.co.jp/kojin/takarakuji/xml/takara_chusen_05.xml"
SMBC_MINI_LOTO_XML_URL = "https://www.smbc.co.jp/kojin/takarakuji/xml/takara_chusen_04.xml"


class SMBCResultSource(CollectorInterface):
    """Collect recent public LOTO6/Mini Loto results from SMBC's result XML."""

    def __init__(
        self,
        *,
        start_date: date = DEFAULT_HISTORY_START,
        end_date: date | None = None,
        minimum_draw_number: int | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date or date.today()
        self.minimum_draw_number = minimum_draw_number
        self.timeout_seconds = timeout_seconds

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        url = _smbc_xml_url(lottery)
        document = fetch_mizuho_document(url, self.timeout_seconds)
        draws = parse_smbc_result_xml(document.text, lottery, document)
        return filter_collected_draws(
            draws,
            lottery,
            self.start_date,
            self.end_date,
            self.minimum_draw_number,
        )


def parse_smbc_result_xml(
    xml_text: str,
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> tuple[HistoricalDraw, ...]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ResearchValidationError(f"invalid SMBC result XML: {exc}") from exc

    draws: list[HistoricalDraw] = []
    for element in root.findall("data"):
        if element.attrib.get("GAME_TYPE") != _smbc_game_type(lottery):
            continue
        draw = _parse_smbc_draw(element.attrib, lottery, document)
        draws.append(draw)
    return tuple(sorted(draws, key=lambda draw: (draw.draw_date, draw.draw_number)))


def _parse_smbc_draw(
    attrs: dict[str, str],
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> HistoricalDraw:
    raw_date = attrs.get("TYUSEN_YMD", "")
    if len(raw_date) != 8 or not raw_date.isdigit():
        raise ResearchValidationError("SMBC result XML row has invalid TYUSEN_YMD")
    draw_date = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))
    draw_number = int(attrs["KAIGOU"])
    main_numbers = _split_fixed_width_numbers(
        attrs.get("TOUSEN_NUMBER", ""),
        lottery.numbers_per_ticket,
    )
    bonus_numbers = _split_fixed_width_numbers(
        attrs.get("BONUS_NUMBER", ""),
        lottery.bonus_numbers,
    )
    return HistoricalDraw(
        lottery=lottery,
        draw_number=draw_number,
        draw_date=draw_date,
        main_numbers=main_numbers,
        bonus_numbers=bonus_numbers,
        source=SMBC_SOURCE,
        source_url=document.url,
        retrieved_at=document.retrieved_at,
        content_hash=document.content_hash,
    )


def _split_fixed_width_numbers(raw_value: str, expected_count: int) -> tuple[int, ...]:
    stripped = raw_value.strip()
    values = tuple(
        int(stripped[index : index + 2])
        for index in range(0, len(stripped), 2)
        if stripped[index : index + 2].strip()
    )
    if len(values) < expected_count:
        raise ResearchValidationError(
            f"SMBC result XML expected {expected_count} numbers, found {len(values)}"
        )
    return values[:expected_count]


def _smbc_xml_url(lottery: LotteryDefinition) -> str:
    if lottery.code == LOTO6.code:
        return SMBC_LOTO6_XML_URL
    if lottery.code == MINI_LOTO.code:
        return SMBC_MINI_LOTO_XML_URL
    raise ResearchValidationError(f"unsupported SMBC result lottery: {lottery.code}")


def _smbc_game_type(lottery: LotteryDefinition) -> str:
    if lottery.code == LOTO6.code:
        return "05"
    if lottery.code == MINI_LOTO.code:
        return "04"
    raise ResearchValidationError(f"unsupported SMBC result lottery: {lottery.code}")
