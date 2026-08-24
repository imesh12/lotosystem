from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import sklearn

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.baseline_benchmark import (
    DEFAULT_STAGE05_SEED,
    DEFAULT_TICKETS_PER_DRAW,
    generate_distinct_random_tickets,
    preflight_validate_benchmark_dataset,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    FEATURE_VERSION_V2,
    LeakageAudit,
    _derived_seed,
    _make_model,
    _scores_from_fitted_model,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.prize import match_ticket
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
    EffectSize,
    PeriodStability,
    bootstrap_confidence_interval,
    classify_conclusion,
    holm_adjust_p_values,
    paired_permutation_p_value,
)

STAGE09_SCHEMA_VERSION = "stage09-portfolio-evaluation-v1"
PORTFOLIO_VERSION = "two-ticket-portfolio-v1"
DEFAULT_CANDIDATE_POOL_SIZE = 50
OVERLAP_PENALTIES = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class ScoredTicket:
    numbers: tuple[int, ...]
    score: float


@dataclass(frozen=True, slots=True)
class Portfolio:
    method: str
    tickets: tuple[tuple[int, ...], tuple[int, ...]]
    ticket_scores: tuple[float, float]
    overlap_count: int
    unique_number_coverage: int
    objective_score: float


@dataclass(frozen=True, slots=True)
class PortfolioConstructionAudit:
    method: str
    deterministic: bool
    tie_breaking: str
    average_overlap: float
    average_unique_number_coverage: float
    top_ranked_numbers_shared: bool


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    draws_evaluated: int
    tickets_evaluated: int
    average_matches_per_ticket: float
    average_matches_per_portfolio: float
    average_best_ticket_matches: float
    match_counts: dict[int, int]
    best_match_count_distribution: dict[int, int]
    match_rates: dict[str, float]
    prize_qualified_rate: float
    portfolio_prize_qualified_rate: float
    average_overlap: float
    average_unique_number_coverage: float


@dataclass(frozen=True, slots=True)
class PortfolioComparison:
    method_value: float
    random_value: float
    difference: float
    difference_ci: ConfidenceInterval
    effect_size: EffectSize
    raw_p_value: float
    adjusted_p_value: float


@dataclass(frozen=True, slots=True)
class PortfolioMethodResult:
    method: str
    configuration: dict[str, Any]
    metrics: PortfolioMetrics
    comparison_vs_random: PortfolioComparison
    period_stability: tuple[PeriodStability, ...]
    conclusion: str


@dataclass(frozen=True, slots=True)
class Stage09PortfolioEvaluationResult:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    feature_version: str
    model_name: str
    feature_group: str
    portfolio_version: str
    configuration: dict[str, Any]
    sklearn_version: str
    current_construction_audit: PortfolioConstructionAudit
    random_metrics: PortfolioMetrics
    method_results: dict[str, PortfolioMethodResult]
    leakage: LeakageAudit
    conclusion: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PortfolioObservation:
    target_draw_date: str
    method_matches_per_portfolio: float
    random_matches_per_portfolio: float
    method_prize_rate: float
    random_prize_rate: float
    portfolio: Portfolio
    random_portfolio: Portfolio
    match_counts: tuple[int, int]
    random_match_counts: tuple[int, int]
    prize_flags: tuple[bool, bool]
    random_prize_flags: tuple[bool, bool]


def run_stage09_portfolio_evaluation(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    ml_refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> Stage09PortfolioEvaluationResult:
    if tickets_per_draw != 2:
        raise ResearchValidationError("Stage 09 evaluates exactly 2 tickets per draw")
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    model_name, feature_group, feature_names, feature_version = _stage08_selected_config(lottery)
    observations = _evaluate_portfolio_methods(
        ordered,
        lottery,
        model_name=model_name,
        feature_names=feature_names,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
        ml_refit_interval=ml_refit_interval,
        candidate_pool_size=candidate_pool_size,
    )
    random_metrics = _portfolio_metrics(next(iter(observations.values())), lottery, use_random=True)
    raw_p_values = {
        method: paired_permutation_p_value(
            tuple(
                observation.method_matches_per_portfolio - observation.random_matches_per_portfolio
                for observation in method_observations
            ),
            seed=_derived_seed(seed, f"{method}-portfolio-permutation"),
            replications=bootstrap_replications,
        )
        for method, method_observations in observations.items()
    }
    adjusted = holm_adjust_p_values(raw_p_values)
    method_results = {
        method: _method_result(
            method,
            method_observations,
            lottery,
            seed=seed,
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
            raw_p_value=raw_p_values[method],
            adjusted_p_value=adjusted[method],
        )
        for method, method_observations in observations.items()
    }
    leakage = _portfolio_leakage_audit(
        ordered,
        lottery,
        feature_names=feature_names,
        model_name=model_name,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 09 leakage audit failed")
    return Stage09PortfolioEvaluationResult(
        schema_version=STAGE09_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        dataset_range={
            "first_draw_number": preflight.first_draw_number,
            "last_draw_number": preflight.last_draw_number,
            "first_draw_date": preflight.first_draw_date,
            "last_draw_date": preflight.last_draw_date,
            "draw_count": preflight.draw_count,
        },
        feature_version=feature_version,
        model_name=model_name,
        feature_group=feature_group,
        portfolio_version=PORTFOLIO_VERSION,
        configuration={
            "seed": seed,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "tickets_per_draw": tickets_per_draw,
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": ml_refit_interval,
            "candidate_pool_size": candidate_pool_size,
            "overlap_penalties": OVERLAP_PENALTIES,
        },
        sklearn_version=sklearn.__version__,
        current_construction_audit=_construction_audit(observations["top_ranked"]),
        random_metrics=random_metrics,
        method_results=method_results,
        leakage=leakage,
        conclusion=_classify_portfolio_conclusion(method_results),
        warnings=(
            "Portfolio objective scores are ranking heuristics, not winning probabilities.",
            "Stage 09 keeps number-scoring models and features unchanged.",
            "No payout amounts, ROI, new model families, or LLM components are included.",
        ),
    )


def build_candidate_pool(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    *,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> tuple[ScoredTicket, ...]:
    ticket_size = lottery.numbers_per_ticket
    top_number_count = min(len(scores), max(ticket_size * 2 + 2, ticket_size + 8))
    top_numbers = tuple(
        number
        for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :top_number_count
        ]
    )
    tickets = [
        ScoredTicket(
            numbers=lottery.validate_main_numbers(numbers),
            score=sum(scores[number] for number in numbers),
        )
        for numbers in combinations(top_numbers, ticket_size)
    ]
    tickets.sort(key=lambda ticket: (-ticket.score, ticket.numbers))
    return tuple(tickets[:candidate_pool_size])


def construct_portfolio(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    method: str,
    *,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> Portfolio:
    if method == "top_ranked":
        ranked = tuple(
            number
            for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        )
        size = lottery.numbers_per_ticket
        tickets = (
            lottery.validate_main_numbers(ranked[:size]),
            lottery.validate_main_numbers(ranked[size : size * 2]),
        )
        return _portfolio(method, tickets, scores, lambda_value=0.0)
    pool = build_candidate_pool(scores, lottery, candidate_pool_size=candidate_pool_size)
    if method == "coverage":
        return _select_pair(pool, method, lambda_value=0.0, coverage_first=True)
    if method == "diversified":
        return _select_pair(pool, method, lambda_value=0.5, coverage_first=False)
    if method.startswith("overlap_penalty_"):
        lambda_value = float(method.removeprefix("overlap_penalty_"))
        return _select_pair(pool, method, lambda_value=lambda_value, coverage_first=False)
    raise ResearchValidationError(f"unknown portfolio method: {method}")


def overlap_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return len(set(left) & set(right))


def unique_coverage(tickets: tuple[tuple[int, ...], tuple[int, ...]]) -> int:
    return len(set(tickets[0]) | set(tickets[1]))


def save_stage09_portfolio_evaluation(
    result: Stage09PortfolioEvaluationResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def _stage08_selected_config(
    lottery: LotteryDefinition,
) -> tuple[str, str, tuple[str, ...], str]:
    if str(lottery.code) == "LOTO6":
        return "random_forest", "gap_only", FEATURE_GROUPS["gap_only"], FEATURE_VERSION_V2
    return "logistic_regression", "pair_only", FEATURE_GROUPS["pair_only"], FEATURE_VERSION_V2


def _evaluate_portfolio_methods(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    model_name: str,
    feature_names: tuple[str, ...],
    seed: int,
    ml_min_training_draws: int,
    ml_refit_interval: int,
    candidate_pool_size: int,
) -> dict[str, tuple[_PortfolioObservation, ...]]:
    methods = (
        "top_ranked",
        "diversified",
        "coverage",
        *(f"overlap_penalty_{value:g}" for value in OVERLAP_PENALTIES),
    )
    observations: dict[str, list[_PortfolioObservation]] = {method: [] for method in methods}
    blocks = build_walk_forward_feature_blocks(draws, lottery, feature_names)
    model = None
    last_fit_index: int | None = None
    for target_index in range(ml_min_training_draws, len(blocks)):
        target_draw = draws[target_index]
        if (
            model is None
            or last_fit_index is None
            or target_index - last_fit_index >= ml_refit_interval
        ):
            x_train, y_train, _training_dates = build_training_dataset(blocks, target_index)
            model = _make_model(model_name, seed)
            model.fit(x_train, y_train)
            last_fit_index = target_index
        scores = _scores_from_fitted_model(model, blocks[target_index])
        random_tickets = generate_distinct_random_tickets(
            lottery,
            random.Random(seed + target_draw.draw_number),
            2,
        )
        random_portfolio = _portfolio("random", random_tickets, scores, lambda_value=0.0)
        random_eval = _evaluate_ticket_pair(random_portfolio, target_draw, lottery)
        for method in methods:
            portfolio = construct_portfolio(
                scores,
                lottery,
                method,
                candidate_pool_size=candidate_pool_size,
            )
            method_eval = _evaluate_ticket_pair(portfolio, target_draw, lottery)
            observations[method].append(
                _PortfolioObservation(
                    target_draw_date=target_draw.draw_date.isoformat(),
                    method_matches_per_portfolio=sum(method_eval["matches"]),
                    random_matches_per_portfolio=sum(random_eval["matches"]),
                    method_prize_rate=sum(method_eval["prizes"]) / 2,
                    random_prize_rate=sum(random_eval["prizes"]) / 2,
                    portfolio=portfolio,
                    random_portfolio=random_portfolio,
                    match_counts=method_eval["matches"],
                    random_match_counts=random_eval["matches"],
                    prize_flags=method_eval["prizes"],
                    random_prize_flags=random_eval["prizes"],
                )
            )
    return {method: tuple(items) for method, items in observations.items()}


def _select_pair(
    pool: tuple[ScoredTicket, ...],
    method: str,
    *,
    lambda_value: float,
    coverage_first: bool,
) -> Portfolio:
    best: tuple[float, float, tuple[int, ...], tuple[int, ...], ScoredTicket, ScoredTicket] | None
    best = None
    for left_index, left in enumerate(pool):
        for right in pool[left_index + 1 :]:
            overlap = overlap_count(left.numbers, right.numbers)
            coverage = len(set(left.numbers) | set(right.numbers))
            objective = left.score + right.score - lambda_value * overlap
            key = (
                float(coverage) if coverage_first else objective,
                objective if coverage_first else float(coverage),
                tuple(-number for number in left.numbers),
                tuple(-number for number in right.numbers),
            )
            comparable = (key[0], key[1], left.numbers, right.numbers, left, right)
            if best is None or comparable > best:
                best = comparable
    if best is None:
        raise ResearchValidationError("candidate pool cannot produce two tickets")
    left = best[4]
    right = best[5]
    tickets = (left.numbers, right.numbers)
    return Portfolio(
        method=method,
        tickets=tickets,
        ticket_scores=(left.score, right.score),
        overlap_count=overlap_count(left.numbers, right.numbers),
        unique_number_coverage=unique_coverage(tickets),
        objective_score=left.score
        + right.score
        - lambda_value * overlap_count(left.numbers, right.numbers),
    )


def _portfolio(
    method: str,
    tickets: tuple[tuple[int, ...], tuple[int, ...]],
    scores: dict[int, float],
    *,
    lambda_value: float,
) -> Portfolio:
    if tickets[0] == tickets[1]:
        raise ResearchValidationError("portfolio tickets must be distinct")
    ticket_scores = tuple(sum(scores.get(number, 0.0) for number in ticket) for ticket in tickets)
    overlap = overlap_count(tickets[0], tickets[1])
    return Portfolio(
        method=method,
        tickets=tickets,
        ticket_scores=(ticket_scores[0], ticket_scores[1]),
        overlap_count=overlap,
        unique_number_coverage=unique_coverage(tickets),
        objective_score=sum(ticket_scores) - lambda_value * overlap,
    )


def _evaluate_ticket_pair(
    portfolio: Portfolio,
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
) -> dict[str, tuple[int, int] | tuple[bool, bool]]:
    results = tuple(match_ticket(ticket, draw, lottery) for ticket in portfolio.tickets)
    return {
        "matches": tuple(result.main_match_count for result in results),
        "prizes": tuple(result.qualifies_for_prize for result in results),
    }


def _portfolio_metrics(
    observations: tuple[_PortfolioObservation, ...],
    lottery: LotteryDefinition,
    *,
    use_random: bool,
) -> PortfolioMetrics:
    matches = tuple(
        match
        for observation in observations
        for match in (observation.random_match_counts if use_random else observation.match_counts)
    )
    prize_flags = tuple(
        flag
        for observation in observations
        for flag in (observation.random_prize_flags if use_random else observation.prize_flags)
    )
    portfolios = tuple(
        observation.random_portfolio if use_random else observation.portfolio
        for observation in observations
    )
    best_counts = tuple(
        max(observation.random_match_counts if use_random else observation.match_counts)
        for observation in observations
    )
    portfolio_prizes = tuple(
        any(observation.random_prize_flags if use_random else observation.prize_flags)
        for observation in observations
    )
    match_counts = {match_count: 0 for match_count in range(lottery.numbers_per_ticket + 1)}
    match_counts.update(Counter(matches))
    best_distribution = {match_count: 0 for match_count in range(lottery.numbers_per_ticket + 1)}
    best_distribution.update(Counter(best_counts))
    return PortfolioMetrics(
        draws_evaluated=len(observations),
        tickets_evaluated=len(matches),
        average_matches_per_ticket=mean(matches) if matches else 0.0,
        average_matches_per_portfolio=mean(
            tuple(
                sum(observation.random_match_counts if use_random else observation.match_counts)
                for observation in observations
            )
        )
        if observations
        else 0.0,
        average_best_ticket_matches=mean(best_counts) if best_counts else 0.0,
        match_counts=dict(sorted(match_counts.items())),
        best_match_count_distribution=dict(sorted(best_distribution.items())),
        match_rates={
            str(match_count): count / len(matches) if matches else 0.0
            for match_count, count in sorted(match_counts.items())
        }
        | {
            "3_plus": _rate(matches, 3),
            "4_plus": _rate(matches, 4),
            "5_plus": _rate(matches, 5),
        },
        prize_qualified_rate=sum(prize_flags) / len(prize_flags) if prize_flags else 0.0,
        portfolio_prize_qualified_rate=sum(portfolio_prizes) / len(portfolio_prizes)
        if portfolio_prizes
        else 0.0,
        average_overlap=mean(tuple(portfolio.overlap_count for portfolio in portfolios))
        if portfolios
        else 0.0,
        average_unique_number_coverage=mean(
            tuple(portfolio.unique_number_coverage for portfolio in portfolios)
        )
        if portfolios
        else 0.0,
    )


def _method_result(
    method: str,
    observations: tuple[_PortfolioObservation, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
    raw_p_value: float,
    adjusted_p_value: float,
) -> PortfolioMethodResult:
    method_values = tuple(observation.method_matches_per_portfolio for observation in observations)
    random_values = tuple(observation.random_matches_per_portfolio for observation in observations)
    differences = tuple(
        left - right for left, right in zip(method_values, random_values, strict=True)
    )
    diff_ci = bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, f"{method}-portfolio-ci"),
        replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    effect = _effect_size(differences, mean(random_values) if random_values else 0.0)
    periods = _period_stability(observations)
    positive_periods = sum(period.mean_match_difference > 0 for period in periods)
    return PortfolioMethodResult(
        method=method,
        configuration=_method_configuration(method),
        metrics=_portfolio_metrics(observations, lottery, use_random=False),
        comparison_vs_random=PortfolioComparison(
            method_value=mean(method_values) if method_values else 0.0,
            random_value=mean(random_values) if random_values else 0.0,
            difference=mean(differences) if differences else 0.0,
            difference_ci=diff_ci,
            effect_size=effect,
            raw_p_value=raw_p_value,
            adjusted_p_value=adjusted_p_value,
        ),
        period_stability=periods,
        conclusion=classify_conclusion(
            adjusted_p_value=adjusted_p_value,
            difference_ci=diff_ci,
            standardized_effect=effect.standardized_mean_difference,
            stable_positive_periods=positive_periods,
            total_periods=len(periods),
        ),
    )


def _method_configuration(method: str) -> dict[str, Any]:
    if method == "top_ranked":
        return {"type": "current", "overlap_penalty": None}
    if method == "coverage":
        return {"type": "coverage_first", "overlap_penalty": None}
    if method == "diversified":
        return {"type": "score_with_overlap_penalty", "overlap_penalty": 0.5}
    if method.startswith("overlap_penalty_"):
        return {
            "type": "score_with_overlap_penalty",
            "overlap_penalty": float(method.removeprefix("overlap_penalty_")),
        }
    return {}


def _construction_audit(
    observations: tuple[_PortfolioObservation, ...],
) -> PortfolioConstructionAudit:
    portfolios = tuple(observation.portfolio for observation in observations)
    return PortfolioConstructionAudit(
        method="top_ranked: Ticket 1 uses the top K scored numbers; Ticket 2 uses the next K.",
        deterministic=True,
        tie_breaking="score descending, then number ascending",
        average_overlap=mean(tuple(portfolio.overlap_count for portfolio in portfolios)),
        average_unique_number_coverage=mean(
            tuple(portfolio.unique_number_coverage for portfolio in portfolios)
        ),
        top_ranked_numbers_shared=False,
    )


def _period_stability(
    observations: tuple[_PortfolioObservation, ...],
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
            observation.method_matches_per_portfolio - observation.random_matches_per_portfolio
            for observation in period_observations
        )
        prize_diff = mean(
            observation.method_prize_rate - observation.random_prize_rate
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


def _portfolio_leakage_audit(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    feature_names: tuple[str, ...],
    model_name: str,
    seed: int,
    ml_min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(ml_min_training_draws, 2), len(draws) - 2)
    original_blocks = build_walk_forward_feature_blocks(draws, lottery, feature_names)
    original_scores = _predict_scores(original_blocks, target_index, model_name, seed)
    original_portfolio = construct_portfolio(original_scores, lottery, "diversified")
    mutated_target = list(draws)
    target_draw = mutated_target[target_index]
    mutated_target[target_index] = HistoricalDraw(
        lottery=target_draw.lottery,
        draw_number=target_draw.draw_number,
        draw_date=target_draw.draw_date,
        main_numbers=tuple(reversed(target_draw.main_numbers)),
        bonus_numbers=target_draw.bonus_numbers,
    )
    target_scores = _predict_scores(
        build_walk_forward_feature_blocks(tuple(mutated_target), lottery, feature_names),
        target_index,
        model_name,
        seed,
    )
    future_mutation = list(draws)
    future_draw = future_mutation[target_index + 1]
    future_mutation[target_index + 1] = HistoricalDraw(
        lottery=future_draw.lottery,
        draw_number=future_draw.draw_number,
        draw_date=future_draw.draw_date,
        main_numbers=tuple(reversed(future_draw.main_numbers)),
        bonus_numbers=future_draw.bonus_numbers,
    )
    future_scores = _predict_scores(
        build_walk_forward_feature_blocks(tuple(future_mutation), lottery, feature_names),
        target_index,
        model_name,
        seed,
    )
    training_dates_ok = all(
        training_date < draws[target_index].draw_date.isoformat()
        for training_date in build_training_dataset(original_blocks, target_index)[2]
    )
    target_changed = original_portfolio != construct_portfolio(
        target_scores, lottery, "diversified"
    )
    future_changed = original_portfolio != construct_portfolio(
        future_scores, lottery, "diversified"
    )
    return LeakageAudit(
        lookahead_safe=training_dates_ok and not target_changed and not future_changed,
        training_dates_strictly_before_target=training_dates_ok,
        target_mutation_changes_features=target_changed,
        future_mutation_changes_prediction=future_changed,
    )


def _predict_scores(
    blocks,
    target_index: int,
    model_name: str,
    seed: int,
) -> dict[int, float]:
    x_train, y_train, _training_dates = build_training_dataset(blocks, target_index)
    model = _make_model(model_name, seed)
    model.fit(x_train, y_train)
    return _scores_from_fitted_model(model, blocks[target_index])


def _classify_portfolio_conclusion(
    results: dict[str, PortfolioMethodResult],
) -> str:
    top = results["top_ranked"]
    best = max(
        results.values(),
        key=lambda result: (
            result.comparison_vs_random.difference,
            result.metrics.average_unique_number_coverage,
        ),
    )
    if best.comparison_vs_random.difference <= 0:
        return (
            "diversification_reduces_overlap_only"
            if best.method != "top_ranked"
            else "no_portfolio_improvement"
        )
    if best.comparison_vs_random.difference <= top.comparison_vs_random.difference:
        return "no_portfolio_improvement"
    if (
        best.comparison_vs_random.difference_ci.lower
        <= 0
        <= best.comparison_vs_random.difference_ci.upper
    ):
        return "weak_portfolio_signal"
    positive_periods = sum(period.mean_match_difference > 0 for period in best.period_stability)
    if positive_periods < max(1, len(best.period_stability) - 1):
        return "unstable_portfolio_signal"
    return "promising_portfolio_method"


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    standard_deviation = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=0.0
        if standard_deviation == 0
        else absolute / standard_deviation,
    )


def _rate(matches: tuple[int, ...], threshold: int) -> float:
    return sum(match >= threshold for match in matches) / len(matches) if matches else 0.0
