from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.baseline_benchmark import (
    DEFAULT_STAGE05_SEED,
    DEFAULT_TICKETS_PER_DRAW,
    _StrategyStatsState,
    generate_distinct_random_tickets,
    preflight_validate_benchmark_dataset,
)
from backend.app.research.candidates import generate_candidates
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.persistence import research_result_json
from backend.app.research.prize import match_ticket

STAGE06_SCHEMA_VERSION = "stage06-statistical-evaluation-v1"
DEFAULT_BOOTSTRAP_REPLICATIONS = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
MULTIPLE_COMPARISON_METHOD = "holm"


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    confidence_level: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class EffectSize:
    absolute_difference: float
    relative_difference: float | None
    standardized_mean_difference: float


@dataclass(frozen=True, slots=True)
class MetricComparison:
    strategy_value: float
    random_value: float
    difference: float
    strategy_ci: ConfidenceInterval
    random_ci: ConfidenceInterval
    difference_ci: ConfidenceInterval
    effect_size: EffectSize
    raw_p_value: float
    adjusted_p_value: float


@dataclass(frozen=True, slots=True)
class PeriodStability:
    period: str
    target_draws: int
    mean_match_difference: float
    prize_rate_difference: float
    direction: str


@dataclass(frozen=True, slots=True)
class StrategyStatisticalEvaluation:
    strategy: str
    experiment_id: str
    target_draws: int
    tickets_per_draw: int
    mean_matches: MetricComparison
    match_rate_3_plus_difference: float
    match_rate_3_plus_ci: ConfidenceInterval
    match_rate_4_plus_difference: float
    match_rate_5_plus_difference: float
    prize_qualified_rate: MetricComparison
    period_stability: tuple[PeriodStability, ...]
    conclusion: str


@dataclass(frozen=True, slots=True)
class Stage06StatisticalEvaluationResult:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    configuration: dict[str, Any]
    multiple_comparison_method: str
    strategies: dict[str, StrategyStatisticalEvaluation]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PairedObservation:
    target_draw_number: int
    target_draw_date: str
    strategy_mean_matches: float
    random_mean_matches: float
    strategy_rate_3_plus: float
    random_rate_3_plus: float
    strategy_rate_4_plus: float
    random_rate_4_plus: float
    strategy_rate_5_plus: float
    random_rate_5_plus: float
    strategy_prize_rate: float
    random_prize_rate: float


def run_stage06_statistical_evaluation(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Stage06StatisticalEvaluationResult:
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")
    if not 0 < confidence_level < 1:
        raise ResearchValidationError("confidence_level must be between 0 and 1")
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    paired = _paired_observations_by_strategy(ordered, lottery, config, tickets_per_draw, seed)

    raw_mean_p_values: dict[str, float] = {}
    raw_prize_p_values: dict[str, float] = {}
    for strategy, observations in paired.items():
        mean_differences = tuple(
            observation.strategy_mean_matches - observation.random_mean_matches
            for observation in observations
        )
        prize_differences = tuple(
            observation.strategy_prize_rate - observation.random_prize_rate
            for observation in observations
        )
        raw_mean_p_values[strategy] = paired_permutation_p_value(
            mean_differences,
            seed=_derived_seed(seed, f"{strategy}-mean-permutation"),
            replications=bootstrap_replications,
        )
        raw_prize_p_values[strategy] = paired_permutation_p_value(
            prize_differences,
            seed=_derived_seed(seed, f"{strategy}-prize-permutation"),
            replications=bootstrap_replications,
        )
    adjusted_mean = holm_adjust_p_values(raw_mean_p_values)
    adjusted_prize = holm_adjust_p_values(raw_prize_p_values)
    final_results = {
        strategy: _evaluate_strategy_observations(
            strategy,
            paired[strategy],
            lottery,
            tickets_per_draw,
            seed,
            bootstrap_replications,
            confidence_level,
            adjusted_mean_p_value=adjusted_mean[strategy],
            adjusted_prize_p_value=adjusted_prize[strategy],
            raw_mean_p_value=raw_mean_p_values[strategy],
            raw_prize_p_value=raw_prize_p_values[strategy],
            dataset_hash=preflight.dataset_hash,
        )
        for strategy in paired
    }
    return Stage06StatisticalEvaluationResult(
        schema_version=STAGE06_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        dataset_range={
            "first_draw_number": preflight.first_draw_number,
            "last_draw_number": preflight.last_draw_number,
            "first_draw_date": preflight.first_draw_date,
            "last_draw_date": preflight.last_draw_date,
            "draw_count": preflight.draw_count,
        },
        configuration={
            "seed": seed,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "tickets_per_draw": tickets_per_draw,
            "backtest_min_training_draws": config.backtest_min_training_draws,
            "frequency_windows": config.frequency_windows,
        },
        multiple_comparison_method=MULTIPLE_COMPARISON_METHOD,
        strategies=final_results,
        warnings=(
            "Statistical detectability does not prove future predictability.",
            "P-values are descriptive research diagnostics and are adjusted across strategies.",
            "Prize payout amounts and ROI are not evaluated.",
        ),
    )


def bootstrap_confidence_interval(
    values: tuple[float, ...],
    *,
    seed: int,
    replications: int,
    confidence_level: float,
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(confidence_level, 0.0, 0.0)
    rng = random.Random(seed)
    sample_size = len(values)
    value_counts = Counter(values)
    unique_values = tuple(value_counts)
    weights = tuple(value_counts[value] for value in unique_values)
    estimates = tuple(
        sum(rng.choices(unique_values, weights=weights, k=sample_size)) / sample_size
        for _ in range(replications)
    )
    lower_tail = (1 - confidence_level) / 2
    upper_tail = 1 - lower_tail
    ordered = tuple(sorted(estimates))
    return ConfidenceInterval(
        confidence_level=confidence_level,
        lower=_quantile(ordered, lower_tail),
        upper=_quantile(ordered, upper_tail),
    )


def paired_permutation_p_value(
    differences: tuple[float, ...],
    *,
    seed: int,
    replications: int,
) -> float:
    if not differences:
        return 1.0
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(replications):
        permuted_mean = mean(
            difference if rng.randrange(2) == 0 else -difference for difference in differences
        )
        if abs(permuted_mean) >= observed:
            extreme += 1
    return (extreme + 1) / (replications + 1)


def holm_adjust_p_values(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        adjusted_value = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted[name] = running_max
    return adjusted


def classify_conclusion(
    *,
    adjusted_p_value: float,
    difference_ci: ConfidenceInterval,
    standardized_effect: float,
    stable_positive_periods: int,
    total_periods: int,
) -> str:
    ci_excludes_zero = difference_ci.lower > 0 or difference_ci.upper < 0
    small_effect = abs(standardized_effect) < 0.1
    mostly_stable = total_periods > 0 and stable_positive_periods >= max(1, total_periods - 1)
    if not ci_excludes_zero and adjusted_p_value >= 0.05:
        return "no_evidence"
    if ci_excludes_zero and adjusted_p_value < 0.05 and small_effect and mostly_stable:
        return "statistically_detectable_small_effect"
    if ci_excludes_zero and not mostly_stable:
        return "unstable_effect"
    if adjusted_p_value < 0.1:
        return "weak_signal"
    return "needs_more_validation"


def deterministic_experiment_id(
    *,
    lottery: str,
    dataset_hash: str,
    strategy: str,
    seed: int,
    bootstrap_replications: int,
    tickets_per_draw: int,
) -> str:
    payload = "|".join(
        (
            STAGE06_SCHEMA_VERSION,
            lottery,
            dataset_hash,
            strategy,
            str(seed),
            str(bootstrap_replications),
            str(tickets_per_draw),
        )
    )
    return "EXP-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_stage06_statistical_evaluation(
    result: Stage06StatisticalEvaluationResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    _write_experiment_records(result, path.parent / "experiments")
    return path


def _paired_observations_by_strategy(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    tickets_per_draw: int,
    seed: int,
) -> dict[str, tuple[_PairedObservation, ...]]:
    strategies = (
        CandidateStrategy.FREQUENCY,
        CandidateStrategy.RECENCY,
        CandidateStrategy.PAIR,
        CandidateStrategy.BALANCED,
        CandidateStrategy.HYBRID,
    )
    observations: dict[str, list[_PairedObservation]] = {
        strategy.value: [] for strategy in strategies
    }
    stats_state = _StrategyStatsState(lottery, config)
    for draw in draws[: config.backtest_min_training_draws]:
        stats_state.add_draw(draw)

    for target_index in range(config.backtest_min_training_draws, len(draws)):
        target_draw = draws[target_index]
        stats = stats_state.to_bundle()
        random_tickets = generate_distinct_random_tickets(
            lottery,
            random.Random(seed + target_draw.draw_number),
            tickets_per_draw,
        )
        random_eval = _evaluate_tickets(random_tickets, target_draw, lottery)
        for strategy in strategies:
            candidates = generate_candidates(
                lottery,
                stats,
                config,
                strategy,
                limit=tickets_per_draw,
            )
            strategy_tickets = tuple(dict.fromkeys(candidate.numbers for candidate in candidates))
            if len(strategy_tickets) < tickets_per_draw:
                raise ResearchValidationError(
                    f"{strategy.value} produced fewer than {tickets_per_draw} distinct "
                    f"candidates for {lottery.code} draw #{target_draw.draw_number}"
                )
            strategy_eval = _evaluate_tickets(
                strategy_tickets[:tickets_per_draw],
                target_draw,
                lottery,
            )
            observations[strategy.value].append(
                _PairedObservation(
                    target_draw_number=target_draw.draw_number,
                    target_draw_date=target_draw.draw_date.isoformat(),
                    strategy_mean_matches=strategy_eval["mean_matches"],
                    random_mean_matches=random_eval["mean_matches"],
                    strategy_rate_3_plus=strategy_eval["rate_3_plus"],
                    random_rate_3_plus=random_eval["rate_3_plus"],
                    strategy_rate_4_plus=strategy_eval["rate_4_plus"],
                    random_rate_4_plus=random_eval["rate_4_plus"],
                    strategy_rate_5_plus=strategy_eval["rate_5_plus"],
                    random_rate_5_plus=random_eval["rate_5_plus"],
                    strategy_prize_rate=strategy_eval["prize_rate"],
                    random_prize_rate=random_eval["prize_rate"],
                )
            )
        stats_state.add_draw(target_draw)
    return {strategy: tuple(values) for strategy, values in observations.items()}


def _evaluate_tickets(
    tickets: tuple[tuple[int, ...], ...],
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
) -> dict[str, float]:
    results = tuple(match_ticket(ticket, draw, lottery) for ticket in tickets)
    matches = tuple(result.main_match_count for result in results)
    return {
        "mean_matches": mean(matches),
        "rate_3_plus": _rate(matches, 3),
        "rate_4_plus": _rate(matches, 4),
        "rate_5_plus": _rate(matches, 5),
        "prize_rate": sum(result.qualifies_for_prize for result in results) / len(results),
    }


def _evaluate_strategy_observations(
    strategy: str,
    observations: tuple[_PairedObservation, ...],
    lottery: LotteryDefinition,
    tickets_per_draw: int,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
    *,
    adjusted_mean_p_value: float,
    adjusted_prize_p_value: float,
    raw_mean_p_value: float,
    raw_prize_p_value: float,
    dataset_hash: str,
) -> StrategyStatisticalEvaluation:
    strategy_means = tuple(observation.strategy_mean_matches for observation in observations)
    random_means = tuple(observation.random_mean_matches for observation in observations)
    mean_differences = tuple(
        left - right for left, right in zip(strategy_means, random_means, strict=True)
    )
    prize_differences = tuple(
        observation.strategy_prize_rate - observation.random_prize_rate
        for observation in observations
    )
    rate_3_differences = tuple(
        observation.strategy_rate_3_plus - observation.random_rate_3_plus
        for observation in observations
    )
    rate_4_differences = tuple(
        observation.strategy_rate_4_plus - observation.random_rate_4_plus
        for observation in observations
    )
    rate_5_differences = tuple(
        observation.strategy_rate_5_plus - observation.random_rate_5_plus
        for observation in observations
    )
    mean_ci = _bootstrap_diff_ci(
        mean_differences, seed, bootstrap_replications, confidence_level, f"{strategy}-mean"
    )
    prize_ci = _bootstrap_diff_ci(
        prize_differences, seed, bootstrap_replications, confidence_level, f"{strategy}-prize"
    )
    periods = _period_stability(observations)
    positive_periods = sum(period.mean_match_difference > 0 for period in periods)
    mean_effect = _effect_size(mean_differences, mean(random_means) if random_means else 0.0)
    conclusion = classify_conclusion(
        adjusted_p_value=adjusted_mean_p_value,
        difference_ci=mean_ci,
        standardized_effect=mean_effect.standardized_mean_difference,
        stable_positive_periods=positive_periods,
        total_periods=len(periods),
    )
    experiment_id = deterministic_experiment_id(
        lottery=str(lottery.code),
        dataset_hash=dataset_hash,
        strategy=strategy,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        tickets_per_draw=tickets_per_draw,
    )
    return StrategyStatisticalEvaluation(
        strategy=strategy,
        experiment_id=experiment_id,
        target_draws=len(observations),
        tickets_per_draw=tickets_per_draw,
        mean_matches=MetricComparison(
            strategy_value=mean(strategy_means) if strategy_means else 0.0,
            random_value=mean(random_means) if random_means else 0.0,
            difference=mean(mean_differences) if mean_differences else 0.0,
            strategy_ci=bootstrap_confidence_interval(
                strategy_means,
                seed=_derived_seed(seed, f"{strategy}-strategy-mean"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            random_ci=bootstrap_confidence_interval(
                random_means,
                seed=_derived_seed(seed, f"{strategy}-random-mean"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            difference_ci=mean_ci,
            effect_size=mean_effect,
            raw_p_value=raw_mean_p_value,
            adjusted_p_value=adjusted_mean_p_value,
        ),
        match_rate_3_plus_difference=mean(rate_3_differences) if rate_3_differences else 0.0,
        match_rate_3_plus_ci=_bootstrap_diff_ci(
            rate_3_differences,
            seed,
            bootstrap_replications,
            confidence_level,
            f"{strategy}-3plus",
        ),
        match_rate_4_plus_difference=mean(rate_4_differences) if rate_4_differences else 0.0,
        match_rate_5_plus_difference=mean(rate_5_differences) if rate_5_differences else 0.0,
        prize_qualified_rate=MetricComparison(
            strategy_value=mean(
                tuple(observation.strategy_prize_rate for observation in observations)
            )
            if observations
            else 0.0,
            random_value=mean(tuple(observation.random_prize_rate for observation in observations))
            if observations
            else 0.0,
            difference=mean(prize_differences) if prize_differences else 0.0,
            strategy_ci=bootstrap_confidence_interval(
                tuple(observation.strategy_prize_rate for observation in observations),
                seed=_derived_seed(seed, f"{strategy}-strategy-prize"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            random_ci=bootstrap_confidence_interval(
                tuple(observation.random_prize_rate for observation in observations),
                seed=_derived_seed(seed, f"{strategy}-random-prize"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            difference_ci=prize_ci,
            effect_size=_effect_size(
                prize_differences,
                mean(tuple(observation.random_prize_rate for observation in observations))
                if observations
                else 0.0,
            ),
            raw_p_value=raw_prize_p_value,
            adjusted_p_value=adjusted_prize_p_value,
        ),
        period_stability=periods,
        conclusion=conclusion,
    )


def _bootstrap_diff_ci(
    differences: tuple[float, ...],
    seed: int,
    replications: int,
    confidence_level: float,
    label: str,
) -> ConfidenceInterval:
    return bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, label),
        replications=replications,
        confidence_level=confidence_level,
    )


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    standard_deviation = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=(
            0.0 if standard_deviation == 0 else absolute / standard_deviation
        ),
    )


def _period_stability(
    observations: tuple[_PairedObservation, ...],
) -> tuple[PeriodStability, ...]:
    periods = (
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024-latest", "2024-01-01", "9999-12-31"),
    )
    results: list[PeriodStability] = []
    for label, start, end in periods:
        period_observations = tuple(
            observation
            for observation in observations
            if start <= observation.target_draw_date <= end
        )
        if not period_observations:
            continue
        mean_diff = mean(
            observation.strategy_mean_matches - observation.random_mean_matches
            for observation in period_observations
        )
        prize_diff = mean(
            observation.strategy_prize_rate - observation.random_prize_rate
            for observation in period_observations
        )
        results.append(
            PeriodStability(
                period=label,
                target_draws=len(period_observations),
                mean_match_difference=mean_diff,
                prize_rate_difference=prize_diff,
                direction="positive" if mean_diff > 0 else "negative" if mean_diff < 0 else "zero",
            )
        )
    return tuple(results)


def _write_experiment_records(
    result: Stage06StatisticalEvaluationResult,
    experiment_dir: Path,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for strategy, evaluation in result.strategies.items():
        payload = {
            "experiment_id": evaluation.experiment_id,
            "lottery": result.lottery,
            "dataset_hash": result.dataset_hash,
            "strategy": strategy,
            "baseline_config": result.configuration,
            "evaluation_range": result.dataset_range,
            "seed": result.configuration["seed"],
            "bootstrap_replications": result.configuration["bootstrap_replications"],
            "metrics": evaluation,
            "confidence_intervals": {
                "mean_difference": evaluation.mean_matches.difference_ci,
                "prize_difference": evaluation.prize_qualified_rate.difference_ci,
            },
            "effect_sizes": {
                "mean_matches": evaluation.mean_matches.effect_size,
                "prize_qualified_rate": evaluation.prize_qualified_rate.effect_size,
            },
            "raw_p_values": {
                "mean_matches": evaluation.mean_matches.raw_p_value,
                "prize_qualified_rate": evaluation.prize_qualified_rate.raw_p_value,
            },
            "adjusted_p_values": {
                "mean_matches": evaluation.mean_matches.adjusted_p_value,
                "prize_qualified_rate": evaluation.prize_qualified_rate.adjusted_p_value,
            },
            "period_stability": evaluation.period_stability,
            "conclusion": evaluation.conclusion,
        }
        path = experiment_dir / f"{evaluation.experiment_id}.json"
        path.write_text(research_result_json(payload), encoding="utf-8")


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _rate(matches: tuple[int, ...], threshold: int) -> float:
    return sum(match >= threshold for match in matches) / len(matches) if matches else 0.0


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return int(digest[:16], 16)
