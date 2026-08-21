from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    EffectSize,
    MetricComparison,
    PeriodStability,
    bootstrap_confidence_interval,
    classify_conclusion,
    holm_adjust_p_values,
    paired_permutation_p_value,
)

STAGE07_SCHEMA_VERSION = "stage07-ml-baseline-v1"
FEATURE_VERSION = "number-features-v1"
DEFAULT_ML_MIN_TRAINING_DRAWS = 100
DEFAULT_ML_REFIT_INTERVAL = 25
DEFAULT_RF_ESTIMATORS = 10
DEFAULT_RF_MAX_DEPTH = 6
DEFAULT_LOGISTIC_MAX_ITER = 250
MODEL_NAMES = ("logistic_regression", "random_forest")


@dataclass(frozen=True, slots=True)
class NumberFeatureRow:
    draw_index: int
    draw_number: int
    draw_date: str
    number: int
    features: tuple[float, ...]
    label: int


@dataclass(frozen=True, slots=True)
class DrawFeatureBlock:
    draw_index: int
    draw_number: int
    draw_date: str
    rows: tuple[NumberFeatureRow, ...]


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    lookahead_safe: bool
    training_dates_strictly_before_target: bool
    target_mutation_changes_features: bool
    future_mutation_changes_prediction: bool


@dataclass(frozen=True, slots=True)
class TicketMetrics:
    draws_evaluated: int
    tickets_evaluated: int
    average_matches_per_ticket: float
    match_counts: dict[int, int]
    match_rates: dict[str, float]
    prize_qualified_rate: float
    prize_category_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class MlModelEvaluation:
    model_name: str
    model_parameters: dict[str, Any]
    mean_matches: MetricComparison
    metrics: TicketMetrics
    period_stability: tuple[PeriodStability, ...]
    conclusion: str


@dataclass(frozen=True, slots=True)
class Stage07MlBaselineResult:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    feature_version: str
    configuration: dict[str, Any]
    sklearn_version: str
    random_metrics: TicketMetrics
    best_deterministic_strategy: str
    best_deterministic_mean_matches: float
    models: dict[str, MlModelEvaluation]
    leakage: LeakageAudit
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DrawOutcome:
    target_draw_number: int
    target_draw_date: str
    tickets: tuple[tuple[int, ...], ...]
    random_tickets: tuple[tuple[int, ...], ...]
    mean_matches: float
    random_mean_matches: float
    prize_rate: float
    random_prize_rate: float
    match_counts: tuple[int, ...]
    random_match_counts: tuple[int, ...]
    prize_names: tuple[str, ...]
    random_prize_names: tuple[str, ...]


class _NumberFeatureState:
    def __init__(self, lottery: LotteryDefinition) -> None:
        self.lottery = lottery
        self.numbers = tuple(range(lottery.number_min, lottery.number_max + 1))
        self.draw_count = 0
        self.total_counts: Counter[int] = Counter()
        self.recent_draws: list[tuple[int, ...]] = []
        self.last_seen_index: dict[int, int] = {}
        self.seen_indices: dict[int, list[int]] = {number: [] for number in self.numbers}
        self.pair_counts: Counter[tuple[int, int]] = Counter()

    def add_draw(self, draw: HistoricalDraw) -> None:
        draw_index = self.draw_count
        self.draw_count += 1
        self.recent_draws.append(draw.main_numbers)
        self.recent_draws = self.recent_draws[-100:]
        for number in draw.main_numbers:
            self.total_counts[number] += 1
            self.last_seen_index[number] = draw_index
            self.seen_indices[number].append(draw_index)
        self.pair_counts.update(combinations(draw.main_numbers, 2))

    def features_for_number(self, number: int) -> tuple[float, ...]:
        frequency_denominator = self.draw_count * self.lottery.numbers_per_ticket
        count = self.total_counts[number]
        gaps = tuple(
            right - left
            for left, right in zip(
                self.seen_indices[number],
                self.seen_indices[number][1:],
                strict=False,
            )
        )
        current_gap = (
            self.draw_count
            if number not in self.last_seen_index
            else self.draw_count - 1 - self.last_seen_index[number]
        )
        pair_strength = sum(
            occurrence for pair, occurrence in self.pair_counts.items() if number in pair
        )
        return (
            count / frequency_denominator if frequency_denominator else 0.0,
            float(self._window_count(number, 5)),
            float(self._window_count(number, 10)),
            float(self._window_count(number, 20)),
            float(self._window_count(number, 50)),
            float(self._window_count(number, 100)),
            float(current_gap),
            mean(gaps) if gaps else 0.0,
            pstdev(gaps) if len(gaps) > 1 else 0.0,
            float(max(gaps) if gaps else 0),
            float(self._window_count(number, 10)),
            pair_strength / self.draw_count if self.draw_count else 0.0,
        )

    def _window_count(self, number: int, window: int) -> int:
        return sum(number in draw_numbers for draw_numbers in self.recent_draws[-window:])


def run_stage07_ml_baseline(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    ml_refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
) -> Stage07MlBaselineResult:
    if tickets_per_draw != 2:
        raise ResearchValidationError("Stage 07 evaluates exactly 2 tickets per draw")
    if ml_min_training_draws < 2:
        raise ResearchValidationError("ml_min_training_draws must be at least 2")
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")
    if ml_refit_interval <= 0:
        raise ResearchValidationError("ml_refit_interval must be positive")

    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough draws for Stage 07 ML walk-forward evaluation")

    blocks = build_walk_forward_feature_blocks(ordered, lottery)
    evaluations: dict[str, tuple[_DrawOutcome, ...]] = {
        model_name: tuple(
            _walk_forward_model_outcomes(
                blocks,
                ordered,
                lottery,
                model_name,
                seed=seed,
                tickets_per_draw=tickets_per_draw,
                ml_min_training_draws=ml_min_training_draws,
                ml_refit_interval=ml_refit_interval,
            )
        )
        for model_name in MODEL_NAMES
    }
    random_metrics = _aggregate_outcomes(
        next(iter(evaluations.values())),
        lottery,
        use_random=True,
    )
    raw_p_values = {
        model_name: paired_permutation_p_value(
            tuple(outcome.mean_matches - outcome.random_mean_matches for outcome in outcomes),
            seed=_derived_seed(seed, f"{model_name}-mean-permutation"),
            replications=bootstrap_replications,
        )
        for model_name, outcomes in evaluations.items()
    }
    adjusted = holm_adjust_p_values(raw_p_values)
    deterministic_name, deterministic_mean = _best_deterministic_strategy(
        ordered,
        lottery,
        config,
        tickets_per_draw,
        ml_min_training_draws,
    )
    model_results = {
        model_name: _model_evaluation(
            model_name,
            outcomes,
            lottery,
            seed=seed,
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
            raw_p_value=raw_p_values[model_name],
            adjusted_p_value=adjusted[model_name],
        )
        for model_name, outcomes in evaluations.items()
    }
    leakage = run_leakage_audit(
        ordered,
        lottery,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 07 leakage audit failed")

    return Stage07MlBaselineResult(
        schema_version=STAGE07_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        dataset_range={
            "first_draw_number": preflight.first_draw_number,
            "last_draw_number": preflight.last_draw_number,
            "first_draw_date": preflight.first_draw_date,
            "last_draw_date": preflight.last_draw_date,
            "draw_count": preflight.draw_count,
        },
        feature_version=FEATURE_VERSION,
        configuration={
            "seed": seed,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "tickets_per_draw": tickets_per_draw,
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": ml_refit_interval,
            "models": _model_parameters(seed),
        },
        sklearn_version=sklearn.__version__,
        random_metrics=random_metrics,
        best_deterministic_strategy=deterministic_name,
        best_deterministic_mean_matches=deterministic_mean,
        models=model_results,
        leakage=leakage,
        warnings=(
            "ML scores are ranking values, not winning probabilities.",
            "Historical ML performance does not guarantee future lottery outcomes.",
            "Bonus numbers are not positive training targets.",
            "Payout amounts and ROI are not evaluated.",
        ),
    )


def build_walk_forward_feature_blocks(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> tuple[DrawFeatureBlock, ...]:
    state = _NumberFeatureState(lottery)
    blocks: list[DrawFeatureBlock] = []
    for draw_index, draw in enumerate(draws):
        rows = tuple(
            NumberFeatureRow(
                draw_index=draw_index,
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                number=number,
                features=state.features_for_number(number),
                label=1 if number in draw.main_numbers else 0,
            )
            for number in state.numbers
        )
        blocks.append(
            DrawFeatureBlock(
                draw_index=draw_index,
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                rows=rows,
            )
        )
        state.add_draw(draw)
    return tuple(blocks)


def build_training_dataset(
    blocks: tuple[DrawFeatureBlock, ...],
    target_index: int,
) -> tuple[list[tuple[float, ...]], list[int], tuple[str, ...]]:
    if target_index <= 1:
        raise ResearchValidationError("target_index must leave earlier training rows")
    training_blocks = blocks[1:target_index]
    x_rows = [row.features for block in training_blocks for row in block.rows]
    y_rows = [row.label for block in training_blocks for row in block.rows]
    training_dates = tuple(block.draw_date for block in training_blocks)
    if len(set(y_rows)) < 2:
        raise ResearchValidationError("training labels require both positive and negative classes")
    return x_rows, y_rows, training_dates


def tickets_from_scores(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    tickets_per_draw: int,
) -> tuple[tuple[int, ...], ...]:
    if tickets_per_draw != 2:
        raise ResearchValidationError("ML baseline currently supports exactly 2 tickets")
    ranked = tuple(
        number for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )
    ticket_size = lottery.numbers_per_ticket
    if len(ranked) < ticket_size * tickets_per_draw:
        raise ResearchValidationError("not enough scored numbers to build two distinct tickets")
    tickets = (
        lottery.validate_main_numbers(ranked[:ticket_size]),
        lottery.validate_main_numbers(ranked[ticket_size : ticket_size * 2]),
    )
    if tickets[0] == tickets[1]:
        raise ResearchValidationError("ML baseline produced duplicate tickets")
    return tickets


def run_leakage_audit(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    ml_min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(ml_min_training_draws, 2), len(draws) - 2)
    original_blocks = build_walk_forward_feature_blocks(draws, lottery)
    target_block = original_blocks[target_index]
    mutated_target = list(draws)
    replacement = tuple(reversed(mutated_target[target_index].main_numbers))
    mutated_target[target_index] = HistoricalDraw(
        lottery=mutated_target[target_index].lottery,
        draw_number=mutated_target[target_index].draw_number,
        draw_date=mutated_target[target_index].draw_date,
        main_numbers=replacement,
        bonus_numbers=mutated_target[target_index].bonus_numbers,
    )
    mutated_target_block = build_walk_forward_feature_blocks(tuple(mutated_target), lottery)[
        target_index
    ]
    future_mutation = list(draws)
    future_draw = future_mutation[target_index + 1]
    future_mutation[target_index + 1] = HistoricalDraw(
        lottery=future_draw.lottery,
        draw_number=future_draw.draw_number,
        draw_date=future_draw.draw_date,
        main_numbers=tuple(reversed(future_draw.main_numbers)),
        bonus_numbers=future_draw.bonus_numbers,
    )
    original_scores = _predict_target_scores(
        original_blocks,
        target_index,
        "logistic_regression",
        seed,
    )
    mutated_future_scores = _predict_target_scores(
        build_walk_forward_feature_blocks(tuple(future_mutation), lottery),
        target_index,
        "logistic_regression",
        seed,
    )
    training_dates_ok = all(
        training_date < draws[target_index].draw_date.isoformat()
        for training_date in build_training_dataset(original_blocks, target_index)[2]
    )
    target_features_changed = tuple(row.features for row in target_block.rows) != tuple(
        row.features for row in mutated_target_block.rows
    )
    future_prediction_changed = original_scores != mutated_future_scores
    return LeakageAudit(
        lookahead_safe=training_dates_ok
        and not target_features_changed
        and not future_prediction_changed,
        training_dates_strictly_before_target=training_dates_ok,
        target_mutation_changes_features=target_features_changed,
        future_mutation_changes_prediction=future_prediction_changed,
    )


def save_stage07_ml_baseline(result: Stage07MlBaselineResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def _walk_forward_model_outcomes(
    blocks: tuple[DrawFeatureBlock, ...],
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    model_name: str,
    *,
    seed: int,
    tickets_per_draw: int,
    ml_min_training_draws: int,
    ml_refit_interval: int,
) -> tuple[_DrawOutcome, ...]:
    outcomes: list[_DrawOutcome] = []
    model: LogisticRegression | RandomForestClassifier | None = None
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
        tickets = tickets_from_scores(scores, lottery, tickets_per_draw)
        random_tickets = generate_distinct_random_tickets(
            lottery,
            random.Random(seed + target_draw.draw_number),
            tickets_per_draw,
        )
        outcomes.append(
            _evaluate_draw_outcome(
                target_draw,
                lottery,
                tickets=tickets,
                random_tickets=random_tickets,
            )
        )
    return tuple(outcomes)


def _predict_target_scores(
    blocks: tuple[DrawFeatureBlock, ...],
    target_index: int,
    model_name: str,
    seed: int,
) -> dict[int, float]:
    x_train, y_train, _training_dates = build_training_dataset(blocks, target_index)
    target_block = blocks[target_index]
    model = _make_model(model_name, seed)
    model.fit(x_train, y_train)
    return _scores_from_fitted_model(model, target_block)


def _scores_from_fitted_model(
    model: LogisticRegression | RandomForestClassifier,
    target_block: DrawFeatureBlock,
) -> dict[int, float]:
    probabilities = model.predict_proba([row.features for row in target_block.rows])
    class_index = list(model.classes_).index(1)
    return {
        row.number: float(probabilities[index][class_index])
        for index, row in enumerate(target_block.rows)
    }


def _make_model(model_name: str, seed: int) -> LogisticRegression | RandomForestClassifier:
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=DEFAULT_LOGISTIC_MAX_ITER,
            random_state=seed,
            solver="liblinear",
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=DEFAULT_RF_ESTIMATORS,
            max_depth=DEFAULT_RF_MAX_DEPTH,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise ResearchValidationError(f"unsupported ML model: {model_name}")


def _model_parameters(seed: int) -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "class_weight": "balanced",
            "max_iter": DEFAULT_LOGISTIC_MAX_ITER,
            "random_state": seed,
            "solver": "liblinear",
        },
        "random_forest": {
            "n_estimators": DEFAULT_RF_ESTIMATORS,
            "max_depth": DEFAULT_RF_MAX_DEPTH,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "random_state": seed,
            "n_jobs": 1,
        },
    }


def _evaluate_draw_outcome(
    target_draw: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    tickets: tuple[tuple[int, ...], ...],
    random_tickets: tuple[tuple[int, ...], ...],
) -> _DrawOutcome:
    model_results = tuple(match_ticket(ticket, target_draw, lottery) for ticket in tickets)
    random_results = tuple(match_ticket(ticket, target_draw, lottery) for ticket in random_tickets)
    model_matches = tuple(result.main_match_count for result in model_results)
    random_matches = tuple(result.main_match_count for result in random_results)
    return _DrawOutcome(
        target_draw_number=target_draw.draw_number,
        target_draw_date=target_draw.draw_date.isoformat(),
        tickets=tickets,
        random_tickets=random_tickets,
        mean_matches=mean(model_matches),
        random_mean_matches=mean(random_matches),
        prize_rate=sum(result.qualifies_for_prize for result in model_results) / len(model_results),
        random_prize_rate=sum(result.qualifies_for_prize for result in random_results)
        / len(random_results),
        match_counts=model_matches,
        random_match_counts=random_matches,
        prize_names=tuple(
            result.prize_name for result in model_results if result.prize_name is not None
        ),
        random_prize_names=tuple(
            result.prize_name for result in random_results if result.prize_name is not None
        ),
    )


def _model_evaluation(
    model_name: str,
    outcomes: tuple[_DrawOutcome, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
    raw_p_value: float,
    adjusted_p_value: float,
) -> MlModelEvaluation:
    model_means = tuple(outcome.mean_matches for outcome in outcomes)
    random_means = tuple(outcome.random_mean_matches for outcome in outcomes)
    differences = tuple(left - right for left, right in zip(model_means, random_means, strict=True))
    periods = _period_stability(outcomes)
    positive_periods = sum(period.mean_match_difference > 0 for period in periods)
    effect = _effect_size(differences, mean(random_means) if random_means else 0.0)
    difference_ci = bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, f"{model_name}-difference-ci"),
        replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    return MlModelEvaluation(
        model_name=model_name,
        model_parameters=_model_parameters(seed)[model_name],
        mean_matches=MetricComparison(
            strategy_value=mean(model_means) if model_means else 0.0,
            random_value=mean(random_means) if random_means else 0.0,
            difference=mean(differences) if differences else 0.0,
            strategy_ci=bootstrap_confidence_interval(
                model_means,
                seed=_derived_seed(seed, f"{model_name}-model-mean-ci"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            random_ci=bootstrap_confidence_interval(
                random_means,
                seed=_derived_seed(seed, f"{model_name}-random-mean-ci"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            ),
            difference_ci=difference_ci,
            effect_size=effect,
            raw_p_value=raw_p_value,
            adjusted_p_value=adjusted_p_value,
        ),
        metrics=_aggregate_outcomes(outcomes, lottery, use_random=False),
        period_stability=periods,
        conclusion=classify_conclusion(
            adjusted_p_value=adjusted_p_value,
            difference_ci=difference_ci,
            standardized_effect=effect.standardized_mean_difference,
            stable_positive_periods=positive_periods,
            total_periods=len(periods),
        ),
    )


def _aggregate_outcomes(
    outcomes: tuple[_DrawOutcome, ...],
    lottery: LotteryDefinition,
    *,
    use_random: bool,
) -> TicketMetrics:
    all_matches = tuple(
        match_count
        for outcome in outcomes
        for match_count in (outcome.random_match_counts if use_random else outcome.match_counts)
    )
    prize_names = tuple(
        prize_name
        for outcome in outcomes
        for prize_name in (outcome.random_prize_names if use_random else outcome.prize_names)
    )
    prize_rate_values = tuple(
        outcome.random_prize_rate if use_random else outcome.prize_rate for outcome in outcomes
    )
    match_counts = {match_count: 0 for match_count in range(lottery.numbers_per_ticket + 1)}
    match_counts.update(Counter(all_matches))
    total = len(all_matches)
    return TicketMetrics(
        draws_evaluated=len(outcomes),
        tickets_evaluated=total,
        average_matches_per_ticket=mean(all_matches) if all_matches else 0.0,
        match_counts=dict(sorted(match_counts.items())),
        match_rates={
            **{
                str(match_count): count / total if total else 0.0
                for match_count, count in sorted(match_counts.items())
            },
            "3_plus": _rate(all_matches, 3),
            "4_plus": _rate(all_matches, 4),
            "5_plus": _rate(all_matches, 5),
        },
        prize_qualified_rate=mean(prize_rate_values) if prize_rate_values else 0.0,
        prize_category_counts=dict(sorted(Counter(prize_names).items())),
    )


def _best_deterministic_strategy(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    tickets_per_draw: int,
    ml_min_training_draws: int,
) -> tuple[str, float]:
    strategies = (
        CandidateStrategy.FREQUENCY,
        CandidateStrategy.RECENCY,
        CandidateStrategy.PAIR,
        CandidateStrategy.BALANCED,
        CandidateStrategy.HYBRID,
    )
    matches: dict[str, list[int]] = {strategy.value: [] for strategy in strategies}
    stats_state = _StrategyStatsState(lottery, config)
    for draw in draws[:ml_min_training_draws]:
        stats_state.add_draw(draw)
    for target_index in range(ml_min_training_draws, len(draws)):
        target_draw = draws[target_index]
        stats = stats_state.to_bundle()
        for strategy in strategies:
            candidates = generate_candidates(
                lottery,
                stats,
                config,
                strategy,
                limit=tickets_per_draw,
            )
            tickets = tuple(dict.fromkeys(candidate.numbers for candidate in candidates))
            if len(tickets) < tickets_per_draw:
                raise ResearchValidationError(
                    f"{strategy.value} produced fewer than {tickets_per_draw} distinct candidates"
                )
            for ticket in tickets[:tickets_per_draw]:
                matches[strategy.value].append(
                    match_ticket(ticket, target_draw, lottery).main_match_count
                )
        stats_state.add_draw(target_draw)
    means = {
        strategy: mean(strategy_matches) if strategy_matches else 0.0
        for strategy, strategy_matches in matches.items()
    }
    return max(means.items(), key=lambda item: (item[1], item[0]))


def _period_stability(outcomes: tuple[_DrawOutcome, ...]) -> tuple[PeriodStability, ...]:
    periods = (
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024-latest", "2024-01-01", "9999-12-31"),
    )
    results: list[PeriodStability] = []
    for label, start, end in periods:
        period_outcomes = tuple(
            outcome for outcome in outcomes if start <= outcome.target_draw_date <= end
        )
        if not period_outcomes:
            continue
        mean_diff = mean(
            outcome.mean_matches - outcome.random_mean_matches for outcome in period_outcomes
        )
        prize_diff = mean(
            outcome.prize_rate - outcome.random_prize_rate for outcome in period_outcomes
        )
        results.append(
            PeriodStability(
                period=label,
                target_draws=len(period_outcomes),
                mean_match_difference=mean_diff,
                prize_rate_difference=prize_diff,
                direction="positive" if mean_diff > 0 else "negative" if mean_diff < 0 else "zero",
            )
        )
    return tuple(results)


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


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return int(digest[:16], 16)
