from __future__ import annotations

import hashlib

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw, validate_draw_sequence
from backend.app.research.exceptions import ResearchValidationError


def validate_lottery_dataset(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    ordered = validate_draw_sequence(draws)
    mismatches = [
        f"draw {draw.draw_number}: {draw.lottery.code}"
        for draw in ordered
        if draw.lottery.code != lottery.code
    ]
    if mismatches:
        raise ResearchValidationError(
            f"mixed-lottery dataset for requested {lottery.code}: " + ", ".join(mismatches)
        )
    return ordered


def calculate_dataset_hash(draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw]) -> str:
    ordered = tuple(
        sorted(draws, key=lambda draw: (str(draw.lottery.code), draw.draw_date, draw.draw_number))
    )
    digest = hashlib.sha256()
    for draw in ordered:
        digest.update(draw.canonical_identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
