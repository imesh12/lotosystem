from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition, PrizeTier
from backend.app.research.baselines import generate_uniform_random_ticket
from backend.app.research.candidates import generate_candidates
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.history_import import find_missing_draw_numbers
from backend.app.research.metrics import BacktestMetrics, calculate_match_metrics
from backend.app.research.persistence import research_result_json
from backend.app.research.prize import match_ticket
from backend.app.research.statistics import (
    FrequencyRecord,
    GapRecord,
    PairRecord,
    PatternStatistics,
    RecencyRecord,
    RecencySummary,
    StatisticsBundle,
    SumStatistics,
    TransitionStatistics,
)

STAGE05_SCHEMA_VERSION = "stage05-baseline-benchmark-v1"
DEFAULT_TICKETS_PER_DRAW = 2
DEFAULT_STAGE05_REPLICATIONS = 1000
DEFAULT_STAGE05_SEED = 123456


@dataclass(frozen=True, slots=True)
class PreflightValidation:
    lottery: str
    first_draw_number: int
    last_draw_number: int
    first_draw_date: str
    last_draw_date: str
    draw_count: int
    missing_draw_numbers: tuple[int, ...]
    dataset_hash: str


@dataclass(frozen=True, slots=True)
class TicketAggregateMetrics:
    draws_evaluated: int
    tickets_evaluated: int
    total_main_matches: int
    average_matches_per_ticket: float
    maximum_matches: int
    match_counts: dict[int, int]
    match_rates: dict[str, float]
    prize_qualified_ticket_count: int
    prize_qualified_rate: float
    prize_category_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    percentile_2_5: float
    percentile_97_5: float


@dataclass(frozen=True, slots=True)
class MonteCarloDistribution:
    average_matches: DistributionSummary
    match_rate_3_plus: DistributionSummary
    match_rate_4_plus: DistributionSummary
    match_rate_5_plus: DistributionSummary
    prize_qualified_rate: DistributionSummary


@dataclass(frozen=True, slots=True)
class RandomBaselineBenchmark:
    replications: int
    seed: int
    tickets_per_draw: int
    aggregate_metrics: TicketAggregateMetrics
    distribution: MonteCarloDistribution


@dataclass(frozen=True, slots=True)
class TheoreticalSanityCheck:
    match_probabilities: dict[int, float]
    match_rate_3_plus: float
    match_rate_4_plus: float
    match_rate_5_plus: float
    prize_qualified_rate: float
    expected_average_matches: float
    monte_carlo_average_matches_delta: float
    monte_carlo_prize_rate_delta: float


@dataclass(frozen=True, slots=True)
class StrategyBenchmark:
    strategy: str
    metrics: TicketAggregateMetrics
    average_match_difference_vs_random: float
    prize_rate_difference_vs_random: float
    lookahead_safe: bool


@dataclass(frozen=True, slots=True)
class CostSummary:
    ticket_price_yen: int
    tickets_per_draw: int
    cost_per_draw_yen: int
    total_historical_simulated_cost_yen: int


@dataclass(frozen=True, slots=True)
class Stage05BenchmarkResult:
    schema_version: str
    lottery: str
    dataset: PreflightValidation
    configuration: dict[str, Any]
    theoretical_probabilities: TheoreticalSanityCheck
    random_baseline: RandomBaselineBenchmark
    strategy_metrics: dict[str, StrategyBenchmark]
    strategy_vs_random_differences: dict[str, dict[str, float]]
    cost: CostSummary
    warnings: tuple[str, ...]


def run_stage05_benchmark(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
) -> Stage05BenchmarkResult:
    if tickets_per_draw <= 0:
        raise ResearchValidationError("tickets_per_draw must be positive")
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED

    random_baseline = run_two_ticket_random_baseline(
        ordered,
        lottery,
        seed=seed,
        replications=config.baseline_replications,
        tickets_per_draw=tickets_per_draw,
    )
    theoretical = calculate_theoretical_sanity_check(
        lottery,
        random_baseline.aggregate_metrics.average_matches_per_ticket,
        random_baseline.aggregate_metrics.prize_qualified_rate,
    )
    strategy_metrics = evaluate_stage05_strategies(
        ordered,
        lottery,
        config,
        tickets_per_draw=tickets_per_draw,
        random_average_matches=random_baseline.aggregate_metrics.average_matches_per_ticket,
        random_prize_qualified_rate=random_baseline.aggregate_metrics.prize_qualified_rate,
    )
    differences = {
        name: {
            "average_matches": (
                benchmark.metrics.average_matches_per_ticket
                - random_baseline.aggregate_metrics.average_matches_per_ticket
            ),
            "prize_qualified_rate": (
                benchmark.metrics.prize_qualified_rate
                - random_baseline.aggregate_metrics.prize_qualified_rate
            ),
        }
        for name, benchmark in strategy_metrics.items()
    }
    return Stage05BenchmarkResult(
        schema_version=STAGE05_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset=preflight,
        configuration={
            "seed": seed,
            "baseline_replications": config.baseline_replications,
            "tickets_per_draw": tickets_per_draw,
            "backtest_min_training_draws": config.backtest_min_training_draws,
            "candidate_pool_numbers": config.candidate_pool_numbers,
            "candidate_limit": config.candidate_limit,
            "frequency_windows": config.frequency_windows,
        },
        theoretical_probabilities=theoretical,
        random_baseline=random_baseline,
        strategy_metrics=strategy_metrics,
        strategy_vs_random_differences=differences,
        cost=CostSummary(
            ticket_price_yen=lottery.ticket_price_yen,
            tickets_per_draw=tickets_per_draw,
            cost_per_draw_yen=tickets_per_draw * lottery.ticket_price_yen,
            total_historical_simulated_cost_yen=(
                len(ordered) * tickets_per_draw * lottery.ticket_price_yen
            ),
        ),
        warnings=(
            "Historical strategy performance does not guarantee future lottery outcomes.",
            "Prize categories are counted, but payout amounts and ROI are not calculated.",
        ),
    )


def preflight_validate_benchmark_dataset(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> PreflightValidation:
    ordered = validate_lottery_dataset(draws, lottery)
    if not ordered:
        raise ResearchValidationError(f"no {lottery.code} draw records available for benchmark")
    missing = find_missing_draw_numbers(ordered)
    if missing:
        raise ResearchValidationError(
            f"{lottery.code} benchmark dataset has missing draw numbers: {missing}"
        )
    return PreflightValidation(
        lottery=str(lottery.code),
        first_draw_number=ordered[0].draw_number,
        last_draw_number=ordered[-1].draw_number,
        first_draw_date=ordered[0].draw_date.isoformat(),
        last_draw_date=ordered[-1].draw_date.isoformat(),
        draw_count=len(ordered),
        missing_draw_numbers=missing,
        dataset_hash=calculate_dataset_hash(ordered),
    )


def generate_distinct_random_tickets(
    lottery: LotteryDefinition,
    rng: random.Random,
    tickets_per_draw: int,
) -> tuple[tuple[int, ...], ...]:
    max_combinations = comb(
        lottery.number_max - lottery.number_min + 1,
        lottery.numbers_per_ticket,
    )
    if tickets_per_draw > max_combinations:
        raise ResearchValidationError("tickets_per_draw exceeds possible unique tickets")
    tickets: set[tuple[int, ...]] = set()
    while len(tickets) < tickets_per_draw:
        tickets.add(generate_uniform_random_ticket(lottery, rng))
    return tuple(sorted(tickets))


def run_two_ticket_random_baseline(
    target_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    replications: int,
    tickets_per_draw: int,
) -> RandomBaselineBenchmark:
    if replications <= 0:
        raise ResearchValidationError("replications must be positive")
    replication_metrics: list[TicketAggregateMetrics] = []
    aggregate_matches: list[int] = []
    aggregate_prizes: list[bool] = []
    aggregate_categories: Counter[str] = Counter()

    for replication_index in range(replications):
        rng = random.Random(_replication_seed(seed, replication_index))
        matches: list[int] = []
        prizes: list[bool] = []
        categories: Counter[str] = Counter()
        for draw in target_draws:
            for ticket in generate_distinct_random_tickets(lottery, rng, tickets_per_draw):
                result = match_ticket(ticket, draw, lottery)
                matches.append(result.main_match_count)
                prizes.append(result.qualifies_for_prize)
                if result.prize_name is not None:
                    categories[result.prize_name] += 1
        metrics = _aggregate_ticket_metrics(
            len(target_draws),
            lottery,
            tuple(matches),
            tuple(prizes),
            dict(categories),
        )
        replication_metrics.append(metrics)
        aggregate_matches.extend(matches)
        aggregate_prizes.extend(prizes)
        aggregate_categories.update(categories)

    return RandomBaselineBenchmark(
        replications=replications,
        seed=seed,
        tickets_per_draw=tickets_per_draw,
        aggregate_metrics=_aggregate_ticket_metrics(
            len(target_draws),
            lottery,
            tuple(aggregate_matches),
            tuple(aggregate_prizes),
            dict(aggregate_categories),
        ),
        distribution=_monte_carlo_distribution(replication_metrics),
    )


def evaluate_stage05_strategies(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int,
    random_average_matches: float,
    random_prize_qualified_rate: float,
) -> dict[str, StrategyBenchmark]:
    strategies = (
        CandidateStrategy.FREQUENCY,
        CandidateStrategy.RECENCY,
        CandidateStrategy.PAIR,
        CandidateStrategy.BALANCED,
        CandidateStrategy.HYBRID,
    )
    matches: dict[CandidateStrategy, list[int]] = {strategy: [] for strategy in strategies}
    prizes: dict[CandidateStrategy, list[bool]] = {strategy: [] for strategy in strategies}
    categories: dict[CandidateStrategy, Counter[str]] = {
        strategy: Counter() for strategy in strategies
    }
    draw_counts: dict[CandidateStrategy, int] = {strategy: 0 for strategy in strategies}
    lookahead_safe = True
    stats_state = _StrategyStatsState(lottery, config)
    for draw in draws[: config.backtest_min_training_draws]:
        stats_state.add_draw(draw)

    for target_index in range(config.backtest_min_training_draws, len(draws)):
        training_draws = draws[:target_index]
        target_draw = draws[target_index]
        previous_draw = training_draws[-1]
        lookahead_safe = lookahead_safe and (
            (previous_draw.draw_date, previous_draw.draw_number)
            < (target_draw.draw_date, target_draw.draw_number)
        )
        stats = stats_state.to_bundle()
        for strategy in strategies:
            candidates = generate_candidates(
                lottery,
                stats,
                config,
                strategy,
                limit=tickets_per_draw,
            )
            distinct_candidates = tuple(
                dict.fromkeys(candidate.numbers for candidate in candidates)
            )
            if len(distinct_candidates) < tickets_per_draw:
                raise ResearchValidationError(
                    f"{strategy.value} produced fewer than {tickets_per_draw} distinct "
                    f"candidates for {lottery.code} draw #{target_draw.draw_number}"
                )
            draw_counts[strategy] += 1
            for ticket in distinct_candidates[:tickets_per_draw]:
                result = match_ticket(ticket, target_draw, lottery)
                matches[strategy].append(result.main_match_count)
                prizes[strategy].append(result.qualifies_for_prize)
                if result.prize_name is not None:
                    categories[strategy][result.prize_name] += 1
        stats_state.add_draw(target_draw)

    results: dict[str, StrategyBenchmark] = {}
    for strategy in strategies:
        benchmark = _aggregate_ticket_metrics(
            draw_counts[strategy],
            lottery,
            tuple(matches[strategy]),
            tuple(prizes[strategy]),
            dict(categories[strategy]),
        )
        results[strategy.value] = StrategyBenchmark(
            strategy=strategy.value,
            metrics=benchmark,
            average_match_difference_vs_random=benchmark.average_matches_per_ticket
            - random_average_matches,
            prize_rate_difference_vs_random=benchmark.prize_qualified_rate
            - random_prize_qualified_rate,
            lookahead_safe=lookahead_safe,
        )
    return results


def calculate_theoretical_sanity_check(
    lottery: LotteryDefinition,
    monte_carlo_average_matches: float,
    monte_carlo_prize_rate: float,
) -> TheoreticalSanityCheck:
    probabilities = _match_probabilities(lottery)
    expected_average_matches = lottery.numbers_per_ticket**2 / (
        lottery.number_max - lottery.number_min + 1
    )
    prize_rate = _theoretical_prize_rate(lottery)
    return TheoreticalSanityCheck(
        match_probabilities=probabilities,
        match_rate_3_plus=sum(
            probability for match_count, probability in probabilities.items() if match_count >= 3
        ),
        match_rate_4_plus=sum(
            probability for match_count, probability in probabilities.items() if match_count >= 4
        ),
        match_rate_5_plus=sum(
            probability for match_count, probability in probabilities.items() if match_count >= 5
        ),
        prize_qualified_rate=prize_rate,
        expected_average_matches=expected_average_matches,
        monte_carlo_average_matches_delta=monte_carlo_average_matches - expected_average_matches,
        monte_carlo_prize_rate_delta=monte_carlo_prize_rate - prize_rate,
    )


def save_stage05_benchmark_result(
    result: Stage05BenchmarkResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def _aggregate_ticket_metrics(
    draws_evaluated: int,
    lottery: LotteryDefinition,
    matches: tuple[int, ...],
    prizes: tuple[bool, ...],
    prize_categories: dict[str, int],
) -> TicketAggregateMetrics:
    metrics = calculate_match_metrics(matches, lottery.numbers_per_ticket, prizes)
    match_counts = {match_count: 0 for match_count in range(lottery.numbers_per_ticket + 1)}
    match_counts.update(metrics.match_distribution)
    return TicketAggregateMetrics(
        draws_evaluated=draws_evaluated,
        tickets_evaluated=len(matches),
        total_main_matches=sum(matches),
        average_matches_per_ticket=metrics.average_matches,
        maximum_matches=metrics.maximum_matches,
        match_counts=dict(sorted(match_counts.items())),
        match_rates=_match_rates(metrics),
        prize_qualified_ticket_count=sum(prizes),
        prize_qualified_rate=metrics.prize_qualified_rate,
        prize_category_counts=_complete_prize_category_counts(lottery, prize_categories),
    )


class _StrategyStatsState:
    def __init__(self, lottery: LotteryDefinition, config: ResearchConfig) -> None:
        self.lottery = lottery
        self.config = config
        self.numbers = tuple(range(lottery.number_min, lottery.number_max + 1))
        self.draw_count = 0
        self.total_counts: Counter[int] = Counter()
        self.recent_draws: list[tuple[int, ...]] = []
        self.last_seen_index: dict[int, int] = {}
        self.seen_indices: dict[int, list[int]] = {number: [] for number in self.numbers}
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.sums: list[int] = []

    def add_draw(self, draw: HistoricalDraw) -> None:
        draw_index = self.draw_count
        self.draw_count += 1
        self.recent_draws.append(draw.main_numbers)
        self.recent_draws = self.recent_draws[-self.config.recent_window :]
        self.sums.append(sum(draw.main_numbers))
        for number in draw.main_numbers:
            self.total_counts[number] += 1
            self.last_seen_index[number] = draw_index
            self.seen_indices[number].append(draw_index)
        self.pair_counts.update(combinations(draw.main_numbers, 2))

    def to_bundle(self) -> StatisticsBundle:
        frequency = self._frequency()
        recency = self._recency()
        return StatisticsBundle(
            lottery_code=str(self.lottery.code),
            draw_count=self.draw_count,
            frequency=frequency,
            recency=recency,
            recency_summary=_recency_summary(recency),
            distributions=(),
            pairs=self._pairs(),
            gaps=self._gaps(),
            sum_statistics=self._sum_statistics(),
            pattern_statistics=PatternStatistics(
                odd_even_patterns={},
                low_high_patterns={},
                consecutive_pair_counts={},
                consecutive_group_counts={},
            ),
            transitions=TransitionStatistics(records=(), overlap_distribution={}),
        )

    def _frequency(self) -> dict[int, FrequencyRecord]:
        total_slots = self.draw_count * self.lottery.numbers_per_ticket
        records: dict[int, FrequencyRecord] = {}
        for number in self.numbers:
            window_counts = {
                window: sum(number in draw_numbers for draw_numbers in self._window_draws(window))
                for window in self.config.frequency_windows
            }
            records[number] = FrequencyRecord(
                number=number,
                total_appearances=self.total_counts[number],
                appearance_rate=self.total_counts[number] / total_slots if total_slots else 0.0,
                window_counts=window_counts,
            )
        return records

    def _recency(self) -> dict[int, RecencyRecord]:
        return {
            number: RecencyRecord(
                number=number,
                draws_since_last_seen=(
                    None
                    if number not in self.last_seen_index
                    else self.draw_count - 1 - self.last_seen_index[number]
                ),
                last_seen_draw=None,
                recent_count=sum(number in draw_numbers for draw_numbers in self.recent_draws),
            )
            for number in self.numbers
        }

    def _pairs(self) -> dict[tuple[int, int], PairRecord]:
        return {
            pair: PairRecord(
                pair=pair,
                occurrence_count=count,
                occurrence_rate=count / self.draw_count if self.draw_count else 0.0,
            )
            for pair, count in sorted(self.pair_counts.items())
            if count >= self.config.min_pair_observations
        }

    def _gaps(self) -> dict[int, GapRecord]:
        records: dict[int, GapRecord] = {}
        for number, indices in self.seen_indices.items():
            gaps = tuple(right - left for left, right in zip(indices, indices[1:], strict=False))
            records[number] = GapRecord(
                number=number,
                average_gap=mean(gaps) if gaps else None,
                median_gap=median(gaps) if gaps else None,
                minimum_gap=min(gaps) if gaps else None,
                maximum_gap=max(gaps) if gaps else None,
                current_gap=None if not indices else self.draw_count - 1 - indices[-1],
                gaps=gaps,
            )
        return records

    def _sum_statistics(self) -> SumStatistics | None:
        if not self.sums:
            return None
        return SumStatistics(
            minimum=min(self.sums),
            maximum=max(self.sums),
            average=mean(self.sums),
            median=median(self.sums),
            percentiles={},
            frequencies=dict(sorted(Counter(self.sums).items())),
        )

    def _window_draws(self, window: int) -> tuple[tuple[int, ...], ...]:
        return tuple(self.recent_draws[-window:])


def _recency_summary(recency: dict[int, RecencyRecord]) -> RecencySummary:
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


def _complete_prize_category_counts(
    lottery: LotteryDefinition,
    prize_categories: dict[str, int],
) -> dict[str, int]:
    counts = {tier.name: 0 for tier in lottery.prize_tiers}
    counts.update(prize_categories)
    return counts


def _match_rates(metrics: BacktestMetrics) -> dict[str, float]:
    total = metrics.total_evaluations
    rates = {
        str(match_count): (count / total if total else 0.0)
        for match_count, count in metrics.match_distribution.items()
    }
    rates["3_plus"] = metrics.match_rate_3_plus
    rates["4_plus"] = metrics.match_rate_4_plus
    rates["5_plus"] = metrics.match_rate_5_plus
    return dict(sorted(rates.items()))


def _monte_carlo_distribution(
    replications: list[TicketAggregateMetrics],
) -> MonteCarloDistribution:
    return MonteCarloDistribution(
        average_matches=_summarize(
            tuple(replication.average_matches_per_ticket for replication in replications)
        ),
        match_rate_3_plus=_summarize(
            tuple(replication.match_rates["3_plus"] for replication in replications)
        ),
        match_rate_4_plus=_summarize(
            tuple(replication.match_rates["4_plus"] for replication in replications)
        ),
        match_rate_5_plus=_summarize(
            tuple(replication.match_rates["5_plus"] for replication in replications)
        ),
        prize_qualified_rate=_summarize(
            tuple(replication.prize_qualified_rate for replication in replications)
        ),
    )


def _summarize(values: tuple[float, ...]) -> DistributionSummary:
    if not values:
        return DistributionSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = tuple(sorted(values))
    return DistributionSummary(
        mean=mean(ordered),
        median=median(ordered),
        standard_deviation=pstdev(ordered) if len(ordered) > 1 else 0.0,
        minimum=ordered[0],
        maximum=ordered[-1],
        percentile_2_5=_quantile(ordered, 0.025),
        percentile_97_5=_quantile(ordered, 0.975),
    )


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _match_probabilities(lottery: LotteryDefinition) -> dict[int, float]:
    number_count = lottery.number_max - lottery.number_min + 1
    ticket_size = lottery.numbers_per_ticket
    denominator = comb(number_count, ticket_size)
    return {
        match_count: (
            comb(ticket_size, match_count)
            * comb(number_count - ticket_size, ticket_size - match_count)
            / denominator
        )
        for match_count in range(ticket_size + 1)
    }


def _theoretical_prize_rate(lottery: LotteryDefinition) -> float:
    return sum(_tier_probability(lottery, tier) for tier in lottery.prize_tiers)


def _tier_probability(lottery: LotteryDefinition, tier: PrizeTier) -> float:
    number_count = lottery.number_max - lottery.number_min + 1
    ticket_size = lottery.numbers_per_ticket
    denominator = comb(number_count, ticket_size)
    main_matches = tier.required_main_matches
    same_match_bonus_tier_exists = any(
        other.required_main_matches == main_matches and other.requires_bonus
        for other in lottery.prize_tiers
    )
    if not tier.requires_bonus and not same_match_bonus_tier_exists:
        return (
            comb(ticket_size, main_matches)
            * comb(number_count - ticket_size, ticket_size - main_matches)
            / denominator
        )
    if not tier.requires_bonus:
        return (
            comb(ticket_size, main_matches)
            * comb(number_count - ticket_size - lottery.bonus_numbers, ticket_size - main_matches)
            / denominator
        )
    non_main_needed = ticket_size - main_matches
    if non_main_needed < 1:
        return 0.0
    return (
        comb(ticket_size, main_matches)
        * comb(lottery.bonus_numbers, 1)
        * comb(
            number_count - ticket_size - lottery.bonus_numbers,
            non_main_needed - 1,
        )
        / denominator
    )


def _replication_seed(seed: int, replication_index: int) -> int:
    return seed + replication_index
