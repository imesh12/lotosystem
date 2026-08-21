from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, median, pstdev

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.metrics import BacktestMetrics, calculate_match_metrics
from backend.app.research.prize import match_ticket


@dataclass(frozen=True, slots=True)
class RandomTicketEvaluation:
    ticket: tuple[int, ...]
    match_count: int
    prize_qualified: bool


@dataclass(frozen=True, slots=True)
class RandomBaselineReplication:
    replication_index: int
    seed: int
    evaluations: tuple[RandomTicketEvaluation, ...]
    metrics: BacktestMetrics
    prize_qualified_rate: float


@dataclass(frozen=True, slots=True)
class RandomBaselineSummary:
    replications: tuple[RandomBaselineReplication, ...]
    mean_matches: float
    median_matches: float
    standard_deviation_matches: float
    maximum_matches: int
    match_distribution: dict[int, int]
    match_rate_3_plus: float
    match_rate_4_plus: float
    match_rate_5_plus: float
    prize_qualified_rate: float


def generate_uniform_random_ticket(
    lottery: LotteryDefinition,
    rng: random.Random,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            rng.sample(
                range(lottery.number_min, lottery.number_max + 1),
                lottery.numbers_per_ticket,
            )
        )
    )


def run_random_baseline_replications(
    target_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    config: ResearchConfig,
) -> RandomBaselineSummary:
    replications: list[RandomBaselineReplication] = []
    master_seed = config.seed if config.seed is not None else 0
    for index in range(config.baseline_replications):
        seed = master_seed + index
        rng = random.Random(seed)
        evaluations = tuple(_evaluate_random_ticket(draw, lottery, rng) for draw in target_draws)
        matches = tuple(evaluation.match_count for evaluation in evaluations)
        replications.append(
            RandomBaselineReplication(
                replication_index=index,
                seed=seed,
                evaluations=evaluations,
                metrics=calculate_match_metrics(
                    matches,
                    lottery.numbers_per_ticket,
                    tuple(evaluation.prize_qualified for evaluation in evaluations),
                ),
                prize_qualified_rate=(
                    sum(evaluation.prize_qualified for evaluation in evaluations) / len(evaluations)
                    if evaluations
                    else 0.0
                ),
            )
        )

    all_matches = tuple(
        evaluation.match_count
        for replication in replications
        for evaluation in replication.evaluations
    )
    all_prizes = tuple(
        evaluation.prize_qualified
        for replication in replications
        for evaluation in replication.evaluations
    )
    distribution = {match_count: 0 for match_count in range(lottery.numbers_per_ticket + 1)}
    for match_count in all_matches:
        distribution[match_count] += 1
    total = len(all_matches)
    return RandomBaselineSummary(
        replications=tuple(replications),
        mean_matches=mean(all_matches) if all_matches else 0.0,
        median_matches=median(all_matches) if all_matches else 0.0,
        standard_deviation_matches=pstdev(all_matches) if len(all_matches) > 1 else 0.0,
        maximum_matches=max(all_matches) if all_matches else 0,
        match_distribution=distribution,
        match_rate_3_plus=_match_rate(all_matches, 3),
        match_rate_4_plus=_match_rate(all_matches, 4),
        match_rate_5_plus=_match_rate(all_matches, 5),
        prize_qualified_rate=sum(all_prizes) / total if total else 0.0,
    )


def _evaluate_random_ticket(
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
    rng: random.Random,
) -> RandomTicketEvaluation:
    ticket = generate_uniform_random_ticket(lottery, rng)
    result = match_ticket(ticket, draw, lottery)
    return RandomTicketEvaluation(
        ticket=ticket,
        match_count=result.main_match_count,
        prize_qualified=result.qualifies_for_prize,
    )


def _match_rate(matches: tuple[int, ...], threshold: int) -> float:
    return (
        sum(match_count >= threshold for match_count in matches) / len(matches) if matches else 0.0
    )
