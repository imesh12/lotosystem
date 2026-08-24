from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.mizuho import SourceDocument, fetch_mizuho_document
from backend.app.research.result_sources import (
    SMBC_LOTO6_XML_URL,
    SMBC_MINI_LOTO_XML_URL,
    SMBC_SOURCE,
)

PAYOUT_SOURCE_MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class DrawPayout:
    lottery: str
    draw_number: int
    prize_tier: str
    payout_yen: int
    winners_count: int | None
    source: str
    source_url: str | None
    retrieved_at: str | None


def validate_draw_payout(payout: DrawPayout, lottery: LotteryDefinition) -> DrawPayout:
    prize_tier = normalize_prize_tier(lottery, payout.prize_tier)
    if prize_tier != payout.prize_tier:
        payout = DrawPayout(
            lottery=payout.lottery,
            draw_number=payout.draw_number,
            prize_tier=prize_tier,
            payout_yen=payout.payout_yen,
            winners_count=payout.winners_count,
            source=payout.source,
            source_url=payout.source_url,
            retrieved_at=payout.retrieved_at,
        )
    valid_tiers = {tier.name for tier in lottery.prize_tiers}
    if payout.lottery != str(lottery.code):
        raise ResearchValidationError(
            f"payout lottery mismatch: expected {lottery.code}, found {payout.lottery}"
        )
    if payout.draw_number <= 0:
        raise ResearchValidationError("payout draw_number must be positive")
    if payout.prize_tier not in valid_tiers:
        raise ResearchValidationError(
            f"invalid payout tier for {lottery.code}: {payout.prize_tier}"
        )
    if payout.payout_yen < 0:
        raise ResearchValidationError("payout_yen must be >= 0")
    if payout.winners_count is not None and payout.winners_count < 0:
        raise ResearchValidationError("winners_count must be >= 0")
    return payout


def normalize_prize_tier(lottery: LotteryDefinition, prize_tier: str) -> str:
    cleaned = prize_tier.strip()
    if cleaned.isdigit():
        rank = int(cleaned)
        for tier in lottery.prize_tiers:
            if tier.rank == rank:
                return tier.name
    return cleaned


def collect_smbc_draw_payouts(
    lottery: LotteryDefinition,
    draw_number: int,
) -> tuple[DrawPayout, ...]:
    url = _smbc_payout_url(lottery)
    document = fetch_mizuho_document(url, 15.0)
    return parse_smbc_payout_xml(document.text, lottery, draw_number, document)


def parse_smbc_payout_xml(
    xml_text: str,
    lottery: LotteryDefinition,
    draw_number: int,
    document: SourceDocument,
) -> tuple[DrawPayout, ...]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ResearchValidationError(f"invalid SMBC payout XML: {exc}") from exc
    for element in root.findall("data"):
        if element.attrib.get("GAME_TYPE") != _smbc_game_type(lottery):
            continue
        if int(element.attrib["KAIGOU"]) != draw_number:
            continue
        return tuple(
            validate_draw_payout(payout, lottery)
            for payout in _payouts_from_smbc_attrs(element.attrib, lottery, document)
        )
    return ()


def manual_draw_payout(
    lottery: LotteryDefinition,
    *,
    draw_number: int,
    prize_tier: str,
    payout_yen: int,
    winners_count: int | None = None,
) -> DrawPayout:
    return validate_draw_payout(
        DrawPayout(
            lottery=str(lottery.code),
            draw_number=draw_number,
            prize_tier=normalize_prize_tier(lottery, prize_tier),
            payout_yen=payout_yen,
            winners_count=winners_count,
            source=PAYOUT_SOURCE_MANUAL,
            source_url="manual:cli",
            retrieved_at=datetime.now(UTC).isoformat(),
        ),
        lottery,
    )


def merge_draw_payouts(
    existing: tuple[DrawPayout, ...],
    incoming: tuple[DrawPayout, ...],
    lottery: LotteryDefinition,
) -> tuple[DrawPayout, ...]:
    merged: dict[str, DrawPayout] = {}
    for payout in (*existing, *incoming):
        validated = validate_draw_payout(payout, lottery)
        current = merged.get(validated.prize_tier)
        if current is not None and (
            current.payout_yen != validated.payout_yen
            or current.winners_count != validated.winners_count
        ):
            raise ResearchValidationError(
                f"conflicting payout for {lottery.code} #{validated.draw_number} "
                f"tier {validated.prize_tier}"
            )
        merged[validated.prize_tier] = validated
    return tuple(merged[tier.name] for tier in lottery.prize_tiers if tier.name in merged)


def _payouts_from_smbc_attrs(
    attrs: dict[str, str],
    lottery: LotteryDefinition,
    document: SourceDocument,
) -> tuple[DrawPayout, ...]:
    payouts: list[DrawPayout] = []
    draw_number = int(attrs["KAIGOU"])
    for tier in lottery.prize_tiers:
        winners = _parse_int(attrs.get(f"TOUSEN_KUTI{tier.rank}"))
        amount = _parse_int(attrs.get(f"TOUSEN_KINGAKU{tier.rank}"))
        if amount is None:
            continue
        payouts.append(
            DrawPayout(
                lottery=str(lottery.code),
                draw_number=draw_number,
                prize_tier=tier.name,
                payout_yen=amount,
                winners_count=winners,
                source=SMBC_SOURCE,
                source_url=document.url,
                retrieved_at=document.retrieved_at.isoformat(),
            )
        )
    return tuple(payouts)


def _parse_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value.strip())


def _smbc_payout_url(lottery: LotteryDefinition) -> str:
    if lottery.code == LOTO6.code:
        return SMBC_LOTO6_XML_URL
    if lottery.code == MINI_LOTO.code:
        return SMBC_MINI_LOTO_XML_URL
    raise ResearchValidationError(f"unsupported SMBC payout lottery: {lottery.code}")


def _smbc_game_type(lottery: LotteryDefinition) -> str:
    if lottery.code == LOTO6.code:
        return "05"
    if lottery.code == MINI_LOTO.code:
        return "04"
    raise ResearchValidationError(f"unsupported SMBC payout lottery: {lottery.code}")
