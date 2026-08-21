from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LotteryCode(StrEnum):
    LOTO6 = "LOTO6"
    MINI_LOTO = "MINI_LOTO"


class DrawFrequency(StrEnum):
    WEEKLY = "weekly"
    TWICE_WEEKLY = "twice_weekly"


class PrizeTier(BaseModel):
    rank: int = Field(ge=1)
    name: str
    required_main_matches: int = Field(ge=0)
    requires_bonus: bool = False
    theoretical_payout_yen: int | None = Field(default=None, ge=0)
    odds: str | None = None
    notes: str | None = None

    model_config = ConfigDict(frozen=True)


class LotteryDefinition(BaseModel):
    code: LotteryCode
    name: str
    number_min: int = Field(ge=1)
    number_max: int = Field(ge=1)
    numbers_per_ticket: int = Field(ge=1)
    bonus_numbers: int = Field(default=0, ge=0)
    ticket_price_yen: int = Field(gt=0)
    draw_frequency: DrawFrequency
    draw_schedule: tuple[str, ...]
    prize_tiers: tuple[PrizeTier, ...]
    active: bool = True

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_definition(self) -> Self:
        if self.number_min > self.number_max:
            raise ValueError("number_min must be less than or equal to number_max")

        available_numbers = self.number_max - self.number_min + 1
        if self.numbers_per_ticket > available_numbers:
            raise ValueError("numbers_per_ticket cannot exceed available lottery numbers")

        if self.bonus_numbers > available_numbers - self.numbers_per_ticket:
            raise ValueError("bonus_numbers cannot exceed remaining lottery numbers")

        for tier in self.prize_tiers:
            if tier.required_main_matches > self.numbers_per_ticket:
                raise ValueError("prize tier cannot require more matches than ticket numbers")

        return self

    @property
    def has_bonus(self) -> bool:
        return self.bonus_numbers > 0

    def validate_main_numbers(self, numbers: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        normalized = tuple(sorted(numbers))

        if len(normalized) != self.numbers_per_ticket:
            raise ValueError(f"{self.code} requires {self.numbers_per_ticket} main numbers")

        if len(set(normalized)) != len(normalized):
            raise ValueError("main numbers must be unique")

        invalid_numbers = [
            number for number in normalized if number < self.number_min or number > self.number_max
        ]
        if invalid_numbers:
            raise ValueError(
                f"numbers must be between {self.number_min} and {self.number_max}: "
                f"{invalid_numbers}"
            )

        return normalized

    def validate_bonus_numbers(
        self,
        bonus_numbers: tuple[int, ...] | list[int],
        *,
        main_numbers: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        normalized = tuple(sorted(bonus_numbers))

        if len(normalized) != self.bonus_numbers:
            raise ValueError(f"{self.code} requires {self.bonus_numbers} bonus numbers")

        if len(set(normalized)) != len(normalized):
            raise ValueError("bonus numbers must be unique")

        main_number_set = set(main_numbers)
        for number in normalized:
            if number < self.number_min or number > self.number_max:
                raise ValueError(
                    f"bonus numbers must be between {self.number_min} and {self.number_max}"
                )
            if number in main_number_set:
                raise ValueError("bonus numbers must not duplicate main numbers")

        return normalized

    def canonical_ticket(self, numbers: tuple[int, ...] | list[int]) -> str:
        return "-".join(f"{number:02d}" for number in self.validate_main_numbers(numbers))
