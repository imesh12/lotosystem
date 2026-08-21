from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.statistics import StatisticsBundle


@dataclass(frozen=True, slots=True)
class NumberFeature:
    number: int
    frequency_rate: float
    recent_frequency: int
    draws_since_last_seen: int | None
    average_gap: float | None
    current_gap: int | None


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    numbers: tuple[int, ...]
    frequency_total: int
    recent_frequency_total: int
    recency_total: int
    average_gap_total: float
    pair_strength: int
    odd_count: int
    even_count: int
    low_count: int
    high_count: int
    total_sum: int
    sum_distance_from_historical_median: float
    consecutive_pair_count: int


def build_number_features(stats: StatisticsBundle) -> dict[int, NumberFeature]:
    return {
        number: NumberFeature(
            number=number,
            frequency_rate=stats.frequency[number].appearance_rate,
            recent_frequency=stats.recency[number].recent_count,
            draws_since_last_seen=stats.recency[number].draws_since_last_seen,
            average_gap=stats.gaps[number].average_gap,
            current_gap=stats.gaps[number].current_gap,
        )
        for number in stats.frequency
    }


def build_candidate_features(
    numbers: tuple[int, ...],
    lottery: LotteryDefinition,
    stats: StatisticsBundle,
    config: ResearchConfig,
) -> CandidateFeatures:
    if stats.lottery_code != str(lottery.code):
        raise ResearchValidationError(
            f"statistics for {stats.lottery_code} cannot be used with {lottery.code}"
        )
    normalized = lottery.validate_main_numbers(numbers)
    threshold = config.threshold_for_range(lottery.number_min, lottery.number_max)
    historical_median = stats.sum_statistics.median if stats.sum_statistics else sum(normalized)
    return CandidateFeatures(
        numbers=normalized,
        frequency_total=sum(stats.frequency[number].total_appearances for number in normalized),
        recent_frequency_total=sum(stats.recency[number].recent_count for number in normalized),
        recency_total=sum(
            stats.recency[number].draws_since_last_seen or stats.draw_count for number in normalized
        ),
        average_gap_total=sum(stats.gaps[number].average_gap or 0.0 for number in normalized),
        pair_strength=sum(
            stats.pairs.get(pair).occurrence_count if stats.pairs.get(pair) else 0
            for pair in combinations(normalized, 2)
        ),
        odd_count=sum(number % 2 for number in normalized),
        even_count=sum(number % 2 == 0 for number in normalized),
        low_count=sum(number <= threshold for number in normalized),
        high_count=sum(number > threshold for number in normalized),
        total_sum=sum(normalized),
        sum_distance_from_historical_median=abs(sum(normalized) - historical_median),
        consecutive_pair_count=sum(
            1 for left, right in zip(normalized, normalized[1:], strict=False) if right - left == 1
        ),
    )
