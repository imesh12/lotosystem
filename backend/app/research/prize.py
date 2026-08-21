from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.lottery import LotteryDefinition, PrizeTier
from backend.app.research.data import HistoricalDraw


@dataclass(frozen=True, slots=True)
class PrizeMatchResult:
    main_match_count: int
    bonus_match_count: int
    bonus_match: bool
    prize_tier: PrizeTier | None

    @property
    def qualifies_for_prize(self) -> bool:
        return self.prize_tier is not None

    @property
    def prize_name(self) -> str | None:
        return self.prize_tier.name if self.prize_tier else None


def match_ticket(
    ticket_numbers: tuple[int, ...] | list[int],
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
) -> PrizeMatchResult:
    normalized_ticket = lottery.validate_main_numbers(ticket_numbers)
    winning_numbers = set(draw.main_numbers)
    bonus_numbers = set(draw.bonus_numbers)
    main_match_count = len(set(normalized_ticket) & winning_numbers)
    bonus_match_count = len(set(normalized_ticket) & bonus_numbers)

    for tier in sorted(lottery.prize_tiers, key=lambda prize_tier: prize_tier.rank):
        if main_match_count != tier.required_main_matches:
            continue
        if tier.requires_bonus and bonus_match_count == 0:
            continue
        return PrizeMatchResult(
            main_match_count=main_match_count,
            bonus_match_count=bonus_match_count,
            bonus_match=bonus_match_count > 0,
            prize_tier=tier,
        )

    return PrizeMatchResult(
        main_match_count=main_match_count,
        bonus_match_count=bonus_match_count,
        bonus_match=bonus_match_count > 0,
        prize_tier=None,
    )
