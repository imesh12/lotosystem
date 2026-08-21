from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    total_evaluations: int
    average_matches: float
    maximum_matches: int
    match_distribution: dict[int, int]
    match_rate_3_plus: float
    match_rate_4_plus: float
    match_rate_5_plus: float
    prize_qualified_rate: float


def calculate_match_metrics(
    matches: tuple[int, ...],
    numbers_per_ticket: int,
    prize_qualified: tuple[bool, ...] = (),
) -> BacktestMetrics:
    distribution = {match_count: 0 for match_count in range(numbers_per_ticket + 1)}
    distribution.update(dict(Counter(matches)))
    return BacktestMetrics(
        total_evaluations=len(matches),
        average_matches=sum(matches) / len(matches) if matches else 0.0,
        maximum_matches=max(matches) if matches else 0,
        match_distribution=dict(sorted(distribution.items())),
        match_rate_3_plus=_match_rate(matches, 3),
        match_rate_4_plus=_match_rate(matches, 4),
        match_rate_5_plus=_match_rate(matches, 5),
        prize_qualified_rate=(
            sum(prize_qualified) / len(prize_qualified) if prize_qualified else 0.0
        ),
    )


def _match_rate(matches: tuple[int, ...], threshold: int) -> float:
    return (
        sum(match_count >= threshold for match_count in matches) / len(matches) if matches else 0.0
    )
