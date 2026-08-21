from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, median

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw, validate_draw_sequence
from backend.app.research.dataset import validate_lottery_dataset


@dataclass(frozen=True, slots=True)
class FrequencyRecord:
    number: int
    total_appearances: int
    appearance_rate: float
    window_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class RecencyRecord:
    number: int
    draws_since_last_seen: int | None
    last_seen_draw: int | None
    recent_count: int


@dataclass(frozen=True, slots=True)
class RecencySummary:
    currently_absent_numbers: tuple[int, ...]
    recently_frequent_numbers: tuple[int, ...]
    recently_inactive_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DistributionRecord:
    draw_number: int
    total_sum: int
    mean: float
    minimum: int
    maximum: int
    median: float
    odd_count: int
    even_count: int
    odd_even_pattern: str
    low_count: int
    high_count: int
    low_high_pattern: str
    consecutive_pair_count: int
    consecutive_group_count: int
    max_consecutive_group_length: int


@dataclass(frozen=True, slots=True)
class PairRecord:
    pair: tuple[int, int]
    occurrence_count: int
    occurrence_rate: float


@dataclass(frozen=True, slots=True)
class GapRecord:
    number: int
    average_gap: float | None
    median_gap: float | None
    minimum_gap: int | None
    maximum_gap: int | None
    current_gap: int | None
    gaps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    from_draw: int
    to_draw: int
    overlap_count: int
    repeat_count: int
    new_number_count: int
    entering_numbers: tuple[int, ...]
    leaving_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SumStatistics:
    minimum: int
    maximum: int
    average: float
    median: float
    percentiles: dict[float, float]
    frequencies: dict[int, int]


@dataclass(frozen=True, slots=True)
class PatternStatistics:
    odd_even_patterns: dict[str, int]
    low_high_patterns: dict[str, int]
    consecutive_pair_counts: dict[int, int]
    consecutive_group_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class TransitionStatistics:
    records: tuple[TransitionRecord, ...]
    overlap_distribution: dict[int, int]


@dataclass(frozen=True, slots=True)
class StatisticsBundle:
    lottery_code: str
    draw_count: int
    frequency: dict[int, FrequencyRecord]
    recency: dict[int, RecencyRecord]
    recency_summary: RecencySummary
    distributions: tuple[DistributionRecord, ...]
    pairs: dict[tuple[int, int], PairRecord]
    gaps: dict[int, GapRecord]
    sum_statistics: SumStatistics | None
    pattern_statistics: PatternStatistics
    transitions: TransitionStatistics


def calculate_statistics(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
) -> StatisticsBundle:
    ordered = validate_lottery_dataset(validate_draw_sequence(draws), lottery)
    numbers = tuple(range(lottery.number_min, lottery.number_max + 1))
    frequency = _calculate_frequency(ordered, numbers, config)
    recency = _calculate_recency(ordered, numbers, config)
    recency_summary = _calculate_recency_summary(recency)
    distributions = tuple(_distribution_for_draw(draw, lottery, config) for draw in ordered)
    pairs = _calculate_pairs(ordered, config)
    gaps = _calculate_gaps(ordered, numbers)
    sum_statistics = _calculate_sum_statistics(distributions, config)
    pattern_statistics = _calculate_pattern_statistics(distributions)
    transitions = _calculate_transitions(ordered)
    return StatisticsBundle(
        lottery_code=str(lottery.code),
        draw_count=len(ordered),
        frequency=frequency,
        recency=recency,
        recency_summary=recency_summary,
        distributions=distributions,
        pairs=pairs,
        gaps=gaps,
        sum_statistics=sum_statistics,
        pattern_statistics=pattern_statistics,
        transitions=transitions,
    )


def _calculate_frequency(
    draws: tuple[HistoricalDraw, ...],
    numbers: tuple[int, ...],
    config: ResearchConfig,
) -> dict[int, FrequencyRecord]:
    overall_counts = Counter(number for draw in draws for number in draw.main_numbers)
    total_slots = len(draws) * (len(draws[0].main_numbers) if draws else 0)
    records: dict[int, FrequencyRecord] = {}
    for number in numbers:
        window_counts = {
            window: sum(number in draw.main_numbers for draw in draws[-window:])
            for window in config.frequency_windows
        }
        records[number] = FrequencyRecord(
            number=number,
            total_appearances=overall_counts[number],
            appearance_rate=overall_counts[number] / total_slots if total_slots else 0.0,
            window_counts=window_counts,
        )
    return records


def _calculate_recency(
    draws: tuple[HistoricalDraw, ...],
    numbers: tuple[int, ...],
    config: ResearchConfig,
) -> dict[int, RecencyRecord]:
    records: dict[int, RecencyRecord] = {}
    for number in numbers:
        last_index = next(
            (
                index
                for index in range(len(draws) - 1, -1, -1)
                if number in draws[index].main_numbers
            ),
            None,
        )
        records[number] = RecencyRecord(
            number=number,
            draws_since_last_seen=None if last_index is None else len(draws) - 1 - last_index,
            last_seen_draw=None if last_index is None else draws[last_index].draw_number,
            recent_count=sum(
                number in draw.main_numbers for draw in draws[-config.recent_window :]
            ),
        )
    return records


def _calculate_recency_summary(recency: dict[int, RecencyRecord]) -> RecencySummary:
    max_recent_count = max((record.recent_count for record in recency.values()), default=0)
    return RecencySummary(
        currently_absent_numbers=tuple(
            number
            for number, record in sorted(recency.items())
            if record.draws_since_last_seen is None or record.draws_since_last_seen > 0
        ),
        recently_frequent_numbers=tuple(
            number
            for number, record in sorted(recency.items())
            if max_recent_count > 0 and record.recent_count == max_recent_count
        ),
        recently_inactive_numbers=tuple(
            number for number, record in sorted(recency.items()) if record.recent_count == 0
        ),
    )


def _distribution_for_draw(
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
    config: ResearchConfig,
) -> DistributionRecord:
    threshold = config.threshold_for_range(lottery.number_min, lottery.number_max)
    numbers = draw.main_numbers
    groups = _consecutive_groups(numbers)
    return DistributionRecord(
        draw_number=draw.draw_number,
        total_sum=sum(numbers),
        mean=mean(numbers),
        minimum=min(numbers),
        maximum=max(numbers),
        median=median(numbers),
        odd_count=sum(number % 2 for number in numbers),
        even_count=sum(number % 2 == 0 for number in numbers),
        odd_even_pattern="".join("O" if number % 2 else "E" for number in numbers),
        low_count=sum(number <= threshold for number in numbers),
        high_count=sum(number > threshold for number in numbers),
        low_high_pattern="".join("L" if number <= threshold else "H" for number in numbers),
        consecutive_pair_count=sum(
            1 for left, right in zip(numbers, numbers[1:], strict=False) if right - left == 1
        ),
        consecutive_group_count=sum(1 for group in groups if len(group) > 1),
        max_consecutive_group_length=max((len(group) for group in groups), default=0),
    )


def _consecutive_groups(numbers: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not numbers:
        return ()
    groups: list[list[int]] = [[numbers[0]]]
    for number in numbers[1:]:
        if number - groups[-1][-1] == 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    return tuple(tuple(group) for group in groups)


def _calculate_pairs(
    draws: tuple[HistoricalDraw, ...],
    config: ResearchConfig,
) -> dict[tuple[int, int], PairRecord]:
    counts = Counter(pair for draw in draws for pair in combinations(draw.main_numbers, 2))
    draw_count = len(draws)
    return {
        pair: PairRecord(pair=pair, occurrence_count=count, occurrence_rate=count / draw_count)
        for pair, count in sorted(counts.items())
        if count >= config.min_pair_observations
    }


def _calculate_gaps(
    draws: tuple[HistoricalDraw, ...],
    numbers: tuple[int, ...],
) -> dict[int, GapRecord]:
    records: dict[int, GapRecord] = {}
    for number in numbers:
        seen_indices = [index for index, draw in enumerate(draws) if number in draw.main_numbers]
        gaps = tuple(
            right - left for left, right in zip(seen_indices, seen_indices[1:], strict=False)
        )
        current_gap = None if not seen_indices else len(draws) - 1 - seen_indices[-1]
        records[number] = GapRecord(
            number=number,
            average_gap=mean(gaps) if gaps else None,
            median_gap=median(gaps) if gaps else None,
            minimum_gap=min(gaps) if gaps else None,
            maximum_gap=max(gaps) if gaps else None,
            current_gap=current_gap,
            gaps=gaps,
        )
    return records


def _calculate_sum_statistics(
    distributions: tuple[DistributionRecord, ...],
    config: ResearchConfig,
) -> SumStatistics | None:
    if not distributions:
        return None
    sums = tuple(distribution.total_sum for distribution in distributions)
    return SumStatistics(
        minimum=min(sums),
        maximum=max(sums),
        average=mean(sums),
        median=median(sums),
        percentiles={
            percentile: _percentile(sums, percentile) for percentile in config.sum_percentiles
        },
        frequencies=dict(sorted(Counter(sums).items())),
    )


def _percentile(values: tuple[int, ...], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile without values")
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _calculate_pattern_statistics(
    distributions: tuple[DistributionRecord, ...],
) -> PatternStatistics:
    return PatternStatistics(
        odd_even_patterns=dict(
            sorted(Counter(record.odd_even_pattern for record in distributions).items())
        ),
        low_high_patterns=dict(
            sorted(Counter(record.low_high_pattern for record in distributions).items())
        ),
        consecutive_pair_counts=dict(
            sorted(Counter(record.consecutive_pair_count for record in distributions).items())
        ),
        consecutive_group_counts=dict(
            sorted(Counter(record.consecutive_group_count for record in distributions).items())
        ),
    )


def _calculate_transitions(draws: tuple[HistoricalDraw, ...]) -> TransitionStatistics:
    records: list[TransitionRecord] = []
    for previous, current in zip(draws, draws[1:], strict=False):
        previous_numbers = set(previous.main_numbers)
        current_numbers = set(current.main_numbers)
        overlap = previous_numbers & current_numbers
        records.append(
            TransitionRecord(
                from_draw=previous.draw_number,
                to_draw=current.draw_number,
                overlap_count=len(overlap),
                repeat_count=len(overlap),
                new_number_count=len(current_numbers - previous_numbers),
                entering_numbers=tuple(sorted(current_numbers - previous_numbers)),
                leaving_numbers=tuple(sorted(previous_numbers - current_numbers)),
            )
        )
    return TransitionStatistics(
        records=tuple(records),
        overlap_distribution=dict(
            sorted(Counter(record.overlap_count for record in records).items())
        ),
    )
