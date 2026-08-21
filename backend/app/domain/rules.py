from backend.app.domain.lottery import DrawFrequency, LotteryCode, LotteryDefinition, PrizeTier

LOTO6 = LotteryDefinition(
    code=LotteryCode.LOTO6,
    name="LOTO6",
    number_min=1,
    number_max=43,
    numbers_per_ticket=6,
    bonus_numbers=1,
    ticket_price_yen=200,
    draw_frequency=DrawFrequency.TWICE_WEEKLY,
    draw_schedule=("Monday", "Thursday"),
    prize_tiers=(
        PrizeTier(
            rank=1,
            name="1st",
            required_main_matches=6,
            theoretical_payout_yen=200_000_000,
            odds="1/6,096,454",
        ),
        PrizeTier(
            rank=2,
            name="2nd",
            required_main_matches=5,
            requires_bonus=True,
            theoretical_payout_yen=10_000_000,
            odds="about 1/1,016,076",
        ),
        PrizeTier(
            rank=3,
            name="3rd",
            required_main_matches=5,
            theoretical_payout_yen=300_000,
            odds="about 1/28,224",
        ),
        PrizeTier(
            rank=4,
            name="4th",
            required_main_matches=4,
            theoretical_payout_yen=6_800,
            odds="about 1/610",
        ),
        PrizeTier(
            rank=5,
            name="5th",
            required_main_matches=3,
            theoretical_payout_yen=1_000,
            odds="about 1/39",
            notes="Generally fixed.",
        ),
    ),
)

MINI_LOTO = LotteryDefinition(
    code=LotteryCode.MINI_LOTO,
    name="Mini Loto",
    number_min=1,
    number_max=31,
    numbers_per_ticket=5,
    bonus_numbers=1,
    ticket_price_yen=200,
    draw_frequency=DrawFrequency.WEEKLY,
    draw_schedule=("Tuesday",),
    prize_tiers=(
        PrizeTier(
            rank=1,
            name="1st",
            required_main_matches=5,
            theoretical_payout_yen=10_000_000,
            odds="1/169,911",
        ),
        PrizeTier(
            rank=2,
            name="2nd",
            required_main_matches=4,
            requires_bonus=True,
            theoretical_payout_yen=150_000,
            odds="about 1/33,982",
        ),
        PrizeTier(
            rank=3,
            name="3rd",
            required_main_matches=4,
            theoretical_payout_yen=10_000,
            odds="about 1/1,359",
        ),
        PrizeTier(
            rank=4,
            name="4th",
            required_main_matches=3,
            theoretical_payout_yen=1_000,
            odds="about 1/52",
        ),
    ),
)

SUPPORTED_LOTTERIES: dict[LotteryCode, LotteryDefinition] = {
    LOTO6.code: LOTO6,
    MINI_LOTO.code: MINI_LOTO,
}


def get_lottery_definition(code: LotteryCode | str) -> LotteryDefinition:
    lottery_code = LotteryCode(code)
    return SUPPORTED_LOTTERIES[lottery_code]
