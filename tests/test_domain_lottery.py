from datetime import date

import pytest
from pydantic import ValidationError

from backend.app.domain import (
    LOTO6,
    MINI_LOTO,
    Draw,
    DrawFrequency,
    Experiment,
    ExperimentStatus,
    LotteryCode,
    Prediction,
    Ticket,
    get_lottery_definition,
)


def test_loto6_rules_are_defined() -> None:
    assert LOTO6.code == LotteryCode.LOTO6
    assert LOTO6.number_min == 1
    assert LOTO6.number_max == 43
    assert LOTO6.numbers_per_ticket == 6
    assert LOTO6.bonus_numbers == 1
    assert LOTO6.ticket_price_yen == 200
    assert LOTO6.draw_frequency == DrawFrequency.TWICE_WEEKLY
    assert LOTO6.draw_schedule == ("Monday", "Thursday")
    assert [tier.rank for tier in LOTO6.prize_tiers] == [1, 2, 3, 4, 5]


def test_mini_loto_rules_are_defined() -> None:
    assert MINI_LOTO.code == LotteryCode.MINI_LOTO
    assert MINI_LOTO.number_min == 1
    assert MINI_LOTO.number_max == 31
    assert MINI_LOTO.numbers_per_ticket == 5
    assert MINI_LOTO.bonus_numbers == 1
    assert MINI_LOTO.ticket_price_yen == 200
    assert MINI_LOTO.draw_frequency == DrawFrequency.WEEKLY
    assert MINI_LOTO.draw_schedule == ("Tuesday",)
    assert [tier.rank for tier in MINI_LOTO.prize_tiers] == [1, 2, 3, 4]


def test_get_lottery_definition_returns_supported_rules() -> None:
    assert get_lottery_definition("LOTO6") == LOTO6
    assert get_lottery_definition(LotteryCode.MINI_LOTO) == MINI_LOTO


def test_ticket_numbers_are_sorted_and_canonicalized() -> None:
    ticket = Ticket(lottery=LOTO6, numbers=(42, 3, 24, 8, 31, 15))

    assert ticket.numbers == (3, 8, 15, 24, 31, 42)
    assert ticket.canonical == "03-08-15-24-31-42"


def test_ticket_rejects_wrong_number_count() -> None:
    with pytest.raises(ValidationError, match="requires 6 main numbers"):
        Ticket(lottery=LOTO6, numbers=(1, 2, 3, 4, 5))


def test_ticket_rejects_duplicate_numbers() -> None:
    with pytest.raises(ValidationError, match="main numbers must be unique"):
        Ticket(lottery=MINI_LOTO, numbers=(1, 2, 3, 4, 4))


def test_ticket_rejects_out_of_range_numbers() -> None:
    with pytest.raises(ValidationError, match="between 1 and 31"):
        Ticket(lottery=MINI_LOTO, numbers=(1, 2, 3, 4, 32))


def test_draw_validates_main_and_bonus_numbers() -> None:
    draw = Draw(
        lottery=MINI_LOTO,
        draw_number=1399,
        draw_date=date(2026, 8, 11),
        main_numbers=(31, 1, 2, 3, 4),
        bonus_numbers=(5,),
    )

    assert draw.main_numbers == (1, 2, 3, 4, 31)
    assert draw.bonus_numbers == (5,)
    assert draw.canonical_main_numbers == "01-02-03-04-31"


def test_draw_rejects_bonus_number_that_duplicates_main_number() -> None:
    with pytest.raises(ValidationError, match="must not duplicate main numbers"):
        Draw(
            lottery=LOTO6,
            draw_number=1,
            draw_date=date(2026, 8, 20),
            main_numbers=(1, 2, 3, 4, 5, 6),
            bonus_numbers=(6,),
        )


def test_prediction_tickets_must_match_prediction_lottery() -> None:
    mini_loto_ticket = Ticket(lottery=MINI_LOTO, numbers=(1, 2, 3, 4, 5))

    with pytest.raises(ValidationError, match="tickets must match"):
        Prediction(
            prediction_id="PRED-001",
            lottery_code=LotteryCode.LOTO6,
            tickets=(mini_loto_ticket,),
        )


def test_prediction_accepts_matching_ticket_lottery() -> None:
    ticket = Ticket(lottery=LOTO6, numbers=(1, 2, 3, 4, 5, 6))
    prediction = Prediction(
        prediction_id="PRED-002",
        lottery_code=LotteryCode.LOTO6,
        tickets=(ticket,),
    )

    assert prediction.tickets == (ticket,)


def test_experiment_defaults_to_draft() -> None:
    experiment = Experiment(
        experiment_id="EXP-0001",
        hypothesis="Lottery-number frequency is a testable feature, not a prediction claim.",
        lottery_code=LotteryCode.LOTO6,
    )

    assert experiment.status == ExperimentStatus.DRAFT
