from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class CandidateStrategy(StrEnum):
    FIXED_BASELINE = "fixed-baseline"
    FREQUENCY = "frequency"
    RECENCY = "recency"
    BALANCED = "balanced"
    PAIR = "pair"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    frequency_windows: tuple[int, ...] = (10, 20, 50, 100)
    recent_window: int = 10
    low_high_threshold: int | None = None
    sum_percentiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    min_pair_observations: int = 1
    candidate_pool_numbers: int = 12
    candidate_limit: int = 10
    backtest_min_training_draws: int = 3
    backtest_candidate_count: int = 1
    evaluation_start: date | None = None
    evaluation_end: date | None = None
    baseline_replications: int = 10
    dataset_version: str = "manual-v001"
    seed: int | None = None

    def __post_init__(self) -> None:
        if any(window <= 0 for window in self.frequency_windows):
            raise ValueError("frequency windows must be positive")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.min_pair_observations <= 0:
            raise ValueError("min_pair_observations must be positive")
        if self.candidate_pool_numbers <= 0:
            raise ValueError("candidate_pool_numbers must be positive")
        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if self.backtest_min_training_draws <= 0:
            raise ValueError("backtest_min_training_draws must be positive")
        if self.backtest_candidate_count <= 0:
            raise ValueError("backtest_candidate_count must be positive")
        if self.baseline_replications <= 0:
            raise ValueError("baseline_replications must be positive")
        if (
            self.evaluation_start
            and self.evaluation_end
            and self.evaluation_start > self.evaluation_end
        ):
            raise ValueError("evaluation_start must be on or before evaluation_end")
        if any(percentile < 0 or percentile > 1 for percentile in self.sum_percentiles):
            raise ValueError("sum_percentiles must be between 0 and 1")

    def threshold_for_range(self, number_min: int, number_max: int) -> int:
        return self.low_high_threshold or (number_min + number_max) // 2
