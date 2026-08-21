from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.features import CandidateFeatures, build_candidate_features
from backend.app.research.statistics import StatisticsBundle


@dataclass(frozen=True, slots=True)
class CandidateScore:
    frequency: float
    recency: float
    gap: float
    pair: float
    distribution: float
    pattern: float

    @property
    def total(self) -> float:
        return (
            self.frequency + self.recency + self.gap + self.pair + self.distribution + self.pattern
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    numbers: tuple[int, ...]
    features: CandidateFeatures
    score: CandidateScore
    strategy: CandidateStrategy

    @property
    def canonical(self) -> str:
        return "-".join(f"{number:02d}" for number in self.numbers)


def generate_candidates(
    lottery: LotteryDefinition,
    stats: StatisticsBundle,
    config: ResearchConfig,
    strategy: CandidateStrategy,
    *,
    limit: int | None = None,
) -> tuple[Candidate, ...]:
    if stats.lottery_code != str(lottery.code):
        raise ResearchValidationError(
            f"statistics for {stats.lottery_code} cannot be used with {lottery.code}"
        )
    candidate_limit = limit or config.candidate_limit
    if strategy == CandidateStrategy.FIXED_BASELINE:
        numbers = tuple(range(lottery.number_min, lottery.number_min + lottery.numbers_per_ticket))
        return (candidate_from_numbers(numbers, lottery, stats, config, strategy),)

    pool = _number_pool(lottery, stats, config, strategy)
    raw_combinations = combinations(pool, lottery.numbers_per_ticket)
    candidates = [
        candidate_from_numbers(numbers, lottery, stats, config, strategy)
        for numbers in raw_combinations
    ]
    candidates.sort(key=lambda candidate: (-candidate.score.total, candidate.numbers))
    return tuple(candidates[:candidate_limit])


def candidate_from_numbers(
    numbers: tuple[int, ...],
    lottery: LotteryDefinition,
    stats: StatisticsBundle,
    config: ResearchConfig,
    strategy: CandidateStrategy,
) -> Candidate:
    features = build_candidate_features(numbers, lottery, stats, config)
    return Candidate(
        numbers=features.numbers,
        features=features,
        score=score_candidate(features, lottery),
        strategy=strategy,
    )


def score_candidate(features: CandidateFeatures, lottery: LotteryDefinition) -> CandidateScore:
    ideal_low = lottery.numbers_per_ticket / 2
    ideal_sum_distance = max(lottery.number_max - lottery.number_min, 1)
    return CandidateScore(
        frequency=float(features.frequency_total),
        recency=float(features.recent_frequency_total),
        gap=features.average_gap_total,
        pair=float(features.pair_strength),
        distribution=(
            -abs(features.low_count - ideal_low)
            - abs(features.odd_count - ideal_low)
            - (features.sum_distance_from_historical_median / ideal_sum_distance)
        ),
        pattern=-float(features.consecutive_pair_count),
    )


def _number_pool(
    lottery: LotteryDefinition,
    stats: StatisticsBundle,
    config: ResearchConfig,
    strategy: CandidateStrategy,
) -> tuple[int, ...]:
    numbers = tuple(range(lottery.number_min, lottery.number_max + 1))
    pool_size = max(config.candidate_pool_numbers, lottery.numbers_per_ticket)

    if strategy == CandidateStrategy.FREQUENCY:
        ranked = sorted(
            numbers, key=lambda number: (-stats.frequency[number].total_appearances, number)
        )
    elif strategy == CandidateStrategy.RECENCY:
        ranked = sorted(
            numbers,
            key=lambda number: (
                stats.recency[number].draws_since_last_seen is None,
                stats.recency[number].draws_since_last_seen or stats.draw_count,
                number,
            ),
        )
    elif strategy == CandidateStrategy.PAIR:
        pair_strength = {
            number: sum(
                record.occurrence_count for pair, record in stats.pairs.items() if number in pair
            )
            for number in numbers
        }
        ranked = sorted(numbers, key=lambda number: (-pair_strength[number], number))
    elif strategy == CandidateStrategy.BALANCED:
        midpoint = config.threshold_for_range(lottery.number_min, lottery.number_max)
        lows = [number for number in numbers if number <= midpoint]
        highs = [number for number in numbers if number > midpoint]
        ranked = tuple(value for pair in zip(lows, highs, strict=False) for value in pair)
        ranked += tuple(lows[len(highs) :]) + tuple(highs[len(lows) :])
    else:
        ranked = sorted(
            numbers,
            key=lambda number: (
                -stats.frequency[number].total_appearances,
                -(stats.gaps[number].average_gap or 0.0),
                -stats.recency[number].recent_count,
                number,
            ),
        )
    return tuple(sorted(ranked[:pool_size]))
