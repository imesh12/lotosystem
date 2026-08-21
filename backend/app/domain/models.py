from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.domain.lottery import LotteryCode, LotteryDefinition


class Ticket(BaseModel):
    lottery: LotteryDefinition
    numbers: tuple[int, ...]

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_ticket(self) -> Self:
        normalized_numbers = self.lottery.validate_main_numbers(self.numbers)
        object.__setattr__(self, "numbers", normalized_numbers)
        return self

    @property
    def canonical(self) -> str:
        return self.lottery.canonical_ticket(self.numbers)


class Draw(BaseModel):
    lottery: LotteryDefinition
    draw_number: int = Field(gt=0)
    draw_date: date
    main_numbers: tuple[int, ...]
    bonus_numbers: tuple[int, ...] = ()

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_draw(self) -> Self:
        normalized_main = self.lottery.validate_main_numbers(self.main_numbers)
        normalized_bonus = self.lottery.validate_bonus_numbers(
            self.bonus_numbers,
            main_numbers=normalized_main,
        )
        object.__setattr__(self, "main_numbers", normalized_main)
        object.__setattr__(self, "bonus_numbers", normalized_bonus)
        return self

    @property
    def canonical_main_numbers(self) -> str:
        return self.lottery.canonical_ticket(self.main_numbers)


class Prediction(BaseModel):
    prediction_id: str
    lottery_code: LotteryCode
    target_draw_number: int | None = Field(default=None, gt=0)
    target_draw_date: date | None = None
    tickets: tuple[Ticket, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_prediction(self) -> Self:
        for ticket in self.tickets:
            if ticket.lottery.code != self.lottery_code:
                raise ValueError("prediction tickets must match prediction lottery_code")
        return self


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"


class Experiment(BaseModel):
    experiment_id: str
    hypothesis: str = Field(min_length=1)
    lottery_code: LotteryCode
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    conclusion: str | None = None

    model_config = ConfigDict(frozen=True)
