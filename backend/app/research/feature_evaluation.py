from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    preflight_validate_benchmark_dataset,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    FEATURE_NAMES_V1,
    FEATURE_NAMES_V2,
    FEATURE_VERSION,
    FEATURE_VERSION_V2,
    MODEL_NAMES,
    DrawFeatureBlock,
    LeakageAudit,
    MlModelEvaluation,
    TicketMetrics,
    _aggregate_outcomes,
    _derived_seed,
    _model_evaluation,
    _predict_target_scores,
    _walk_forward_model_outcomes,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    holm_adjust_p_values,
    paired_permutation_p_value,
)

STAGE08_SCHEMA_VERSION = "stage08-feature-evaluation-v1"
CORRELATION_THRESHOLD = 0.95
NEAR_CONSTANT_UNIQUE_RATE = 0.005

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "v1_all": FEATURE_NAMES_V1,
    "frequency_only": (
        "frequency_rate",
        "frequency_5",
        "frequency_10",
        "frequency_20",
        "frequency_50",
        "frequency_100",
        "recent_activity_10",
    ),
    "gap_only": ("current_gap", "mean_gap", "gap_std", "max_gap"),
    "pair_only": ("pair_strength_rate",),
    "v2_all": FEATURE_NAMES_V2,
    "v2_without_frequency_expansion": tuple(
        name for name in FEATURE_NAMES_V2 if not name.startswith("frequency_")
    ),
    "v2_without_gap_expansion": tuple(
        name
        for name in FEATURE_NAMES_V2
        if name not in {"gap_to_mean", "gap_to_median", "gap_z_score"}
    ),
    "v2_without_pair_transition": tuple(
        name
        for name in FEATURE_NAMES_V2
        if name
        not in {
            "pair_mean_strength_rate",
            "pair_max_strength_rate",
            "previous_draw_presence",
            "previous_draw_pair_strength_rate",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class FeatureAuditRecord:
    name: str
    value_type: str
    missing_count: int
    missing_rate: float
    constant: bool
    near_constant: bool
    minimum: float
    maximum: float
    average: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class CorrelatedFeaturePair:
    feature_a: str
    feature_b: str
    correlation: float


@dataclass(frozen=True, slots=True)
class TemporalFeatureShift:
    feature: str
    period_means: dict[str, float]
    maximum_mean_shift: float


@dataclass(frozen=True, slots=True)
class FeatureAudit:
    feature_version: str
    feature_count: int
    records: dict[str, FeatureAuditRecord]
    correlated_pairs: tuple[CorrelatedFeaturePair, ...]
    temporal_shifts: tuple[TemporalFeatureShift, ...]


@dataclass(frozen=True, slots=True)
class AblationResult:
    feature_group: str
    feature_names: tuple[str, ...]
    random_metrics: TicketMetrics
    models: dict[str, MlModelEvaluation]
    best_model: str
    best_mean_matches: float
    best_difference_vs_random: float


@dataclass(frozen=True, slots=True)
class FeatureImportanceResult:
    model_name: str
    top_features: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class Stage08FeatureEvaluationResult:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    configuration: dict[str, Any]
    sklearn_version: str
    feature_groups: dict[str, tuple[str, ...]]
    v1_audit: FeatureAudit
    v2_audit: FeatureAudit
    ablation_results: dict[str, AblationResult]
    feature_importance: dict[str, FeatureImportanceResult]
    stage07_comparison: dict[str, Any]
    leakage: LeakageAudit
    conclusion: str
    warnings: tuple[str, ...]


def run_stage08_feature_evaluation(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    ml_refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
) -> Stage08FeatureEvaluationResult:
    if tickets_per_draw != 2:
        raise ResearchValidationError("Stage 08 evaluates exactly 2 tickets per draw")
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough draws for Stage 08 feature evaluation")

    v1_blocks = build_walk_forward_feature_blocks(ordered, lottery, FEATURE_NAMES_V1)
    v2_blocks = build_walk_forward_feature_blocks(ordered, lottery, FEATURE_NAMES_V2)
    v1_audit = audit_feature_blocks(v1_blocks, FEATURE_NAMES_V1, FEATURE_VERSION)
    v2_audit = audit_feature_blocks(v2_blocks, FEATURE_NAMES_V2, FEATURE_VERSION_V2)
    ablations = {
        group_name: _evaluate_feature_group(
            group_name,
            feature_names,
            ordered,
            lottery,
            seed=seed,
            tickets_per_draw=tickets_per_draw,
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
            ml_min_training_draws=ml_min_training_draws,
            ml_refit_interval=ml_refit_interval,
        )
        for group_name, feature_names in FEATURE_GROUPS.items()
    }
    leakage = run_feature_leakage_audit(
        ordered,
        lottery,
        feature_names=FEATURE_NAMES_V2,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 08 leakage audit failed")
    importance = calculate_feature_importance(
        v2_blocks,
        feature_names=FEATURE_NAMES_V2,
        seed=seed,
        target_index=len(v2_blocks) - 1,
    )
    return Stage08FeatureEvaluationResult(
        schema_version=STAGE08_SCHEMA_VERSION,
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
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": ml_refit_interval,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "near_constant_unique_rate": NEAR_CONSTANT_UNIQUE_RATE,
        },
        sklearn_version=sklearn.__version__,
        feature_groups=FEATURE_GROUPS,
        v1_audit=v1_audit,
        v2_audit=v2_audit,
        ablation_results=ablations,
        feature_importance=importance,
        stage07_comparison=_load_stage07_comparison(str(lottery.code)),
        leakage=leakage,
        conclusion=_classify_feature_conclusion(ablations),
        warnings=(
            "Feature importance is descriptive and does not prove predictive value.",
            "Expanded features are historical-only hypotheses, not winning probabilities.",
            "No payout amounts, ROI, advanced ML models, or LLM components are included.",
        ),
    )


def audit_feature_blocks(
    blocks: tuple[DrawFeatureBlock, ...],
    feature_names: tuple[str, ...],
    feature_version: str,
) -> FeatureAudit:
    values_by_feature = _values_by_feature(blocks, feature_names)
    records = {name: _audit_feature(name, values) for name, values in values_by_feature.items()}
    return FeatureAudit(
        feature_version=feature_version,
        feature_count=len(feature_names),
        records=records,
        correlated_pairs=_correlated_pairs(values_by_feature),
        temporal_shifts=_temporal_shifts(blocks, feature_names),
    )


def calculate_feature_importance(
    blocks: tuple[DrawFeatureBlock, ...],
    *,
    feature_names: tuple[str, ...],
    seed: int,
    target_index: int,
) -> dict[str, FeatureImportanceResult]:
    x_train, y_train, _training_dates = build_training_dataset(blocks, target_index)
    scaled_x, means, stds = _standardize(x_train)
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=250,
        random_state=seed,
        solver="liblinear",
    )
    logistic.fit(scaled_x, y_train)
    forest = RandomForestClassifier(
        n_estimators=10,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    forest.fit(x_train, y_train)
    logistic_scores = tuple(
        (feature_names[index], abs(float(logistic.coef_[0][index])))
        for index in range(len(feature_names))
    )
    forest_scores = tuple(
        (feature_names[index], float(forest.feature_importances_[index]))
        for index in range(len(feature_names))
    )
    return {
        "logistic_regression": FeatureImportanceResult(
            model_name="logistic_regression",
            top_features=_top_features(logistic_scores),
        ),
        "random_forest": FeatureImportanceResult(
            model_name="random_forest",
            top_features=_top_features(forest_scores),
        ),
    }


def run_feature_leakage_audit(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    feature_names: tuple[str, ...],
    seed: int,
    ml_min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(ml_min_training_draws, 2), len(draws) - 2)
    original_blocks = build_walk_forward_feature_blocks(draws, lottery, feature_names)
    target_block = original_blocks[target_index]
    mutated_target = list(draws)
    target_draw = mutated_target[target_index]
    mutated_target[target_index] = HistoricalDraw(
        lottery=target_draw.lottery,
        draw_number=target_draw.draw_number,
        draw_date=target_draw.draw_date,
        main_numbers=tuple(reversed(target_draw.main_numbers)),
        bonus_numbers=target_draw.bonus_numbers,
    )
    mutated_target_block = build_walk_forward_feature_blocks(
        tuple(mutated_target), lottery, feature_names
    )[target_index]
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
    future_scores = _predict_target_scores(
        build_walk_forward_feature_blocks(tuple(future_mutation), lottery, feature_names),
        target_index,
        "logistic_regression",
        seed,
    )
    truncated_scores = _predict_target_scores(
        build_walk_forward_feature_blocks(draws[: target_index + 1], lottery, feature_names),
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
    future_prediction_changed = (
        original_scores != future_scores or original_scores != truncated_scores
    )
    return LeakageAudit(
        lookahead_safe=training_dates_ok
        and not target_features_changed
        and not future_prediction_changed,
        training_dates_strictly_before_target=training_dates_ok,
        target_mutation_changes_features=target_features_changed,
        future_mutation_changes_prediction=future_prediction_changed,
    )


def save_stage08_feature_evaluation(
    result: Stage08FeatureEvaluationResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def _evaluate_feature_group(
    group_name: str,
    feature_names: tuple[str, ...],
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    tickets_per_draw: int,
    bootstrap_replications: int,
    confidence_level: float,
    ml_min_training_draws: int,
    ml_refit_interval: int,
) -> AblationResult:
    blocks = build_walk_forward_feature_blocks(draws, lottery, feature_names)
    outcomes_by_model = {
        model_name: _walk_forward_model_outcomes(
            blocks,
            draws,
            lottery,
            model_name,
            seed=seed,
            tickets_per_draw=tickets_per_draw,
            ml_min_training_draws=ml_min_training_draws,
            ml_refit_interval=ml_refit_interval,
        )
        for model_name in MODEL_NAMES
    }
    raw_p_values = {
        model_name: paired_permutation_p_value(
            tuple(outcome.mean_matches - outcome.random_mean_matches for outcome in outcomes),
            seed=_derived_seed(seed, f"{group_name}-{model_name}-mean-permutation"),
            replications=bootstrap_replications,
        )
        for model_name, outcomes in outcomes_by_model.items()
    }
    adjusted = holm_adjust_p_values(raw_p_values)
    model_results = {
        model_name: _model_evaluation(
            model_name,
            outcomes,
            lottery,
            seed=_derived_seed(seed, group_name),
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
            raw_p_value=raw_p_values[model_name],
            adjusted_p_value=adjusted[model_name],
        )
        for model_name, outcomes in outcomes_by_model.items()
    }
    best_model, best_result = max(
        model_results.items(),
        key=lambda item: (item[1].mean_matches.difference, item[1].mean_matches.strategy_value),
    )
    return AblationResult(
        feature_group=group_name,
        feature_names=feature_names,
        random_metrics=_aggregate_outcomes(
            next(iter(outcomes_by_model.values())), lottery, use_random=True
        ),
        models=model_results,
        best_model=best_model,
        best_mean_matches=best_result.mean_matches.strategy_value,
        best_difference_vs_random=best_result.mean_matches.difference,
    )


def _audit_feature(name: str, values: tuple[float, ...]) -> FeatureAuditRecord:
    finite_values = tuple(value for value in values if math.isfinite(value))
    missing_count = len(values) - len(finite_values)
    unique_rate = len(set(finite_values)) / len(finite_values) if finite_values else 0.0
    return FeatureAuditRecord(
        name=name,
        value_type="float",
        missing_count=missing_count,
        missing_rate=missing_count / len(values) if values else 0.0,
        constant=len(set(finite_values)) <= 1,
        near_constant=unique_rate <= NEAR_CONSTANT_UNIQUE_RATE,
        minimum=min(finite_values) if finite_values else 0.0,
        maximum=max(finite_values) if finite_values else 0.0,
        average=mean(finite_values) if finite_values else 0.0,
        standard_deviation=pstdev(finite_values) if len(finite_values) > 1 else 0.0,
    )


def _values_by_feature(
    blocks: tuple[DrawFeatureBlock, ...],
    feature_names: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    values = {name: [] for name in feature_names}
    for block in blocks[1:]:
        for row in block.rows:
            for index, name in enumerate(feature_names):
                values[name].append(row.features[index])
    return {name: tuple(items) for name, items in values.items()}


def _correlated_pairs(
    values_by_feature: dict[str, tuple[float, ...]],
) -> tuple[CorrelatedFeaturePair, ...]:
    names = tuple(values_by_feature)
    pairs: list[CorrelatedFeaturePair] = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            correlation = _correlation(values_by_feature[left_name], values_by_feature[right_name])
            if abs(correlation) >= CORRELATION_THRESHOLD:
                pairs.append(CorrelatedFeaturePair(left_name, right_name, correlation))
    return tuple(sorted(pairs, key=lambda pair: (-abs(pair.correlation), pair.feature_a)))


def _temporal_shifts(
    blocks: tuple[DrawFeatureBlock, ...],
    feature_names: tuple[str, ...],
) -> tuple[TemporalFeatureShift, ...]:
    periods = (
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024-latest", "2024-01-01", "9999-12-31"),
    )
    shifts: list[TemporalFeatureShift] = []
    for feature_index, feature_name in enumerate(feature_names):
        period_means: dict[str, float] = {}
        for label, start, end in periods:
            values = tuple(
                row.features[feature_index]
                for block in blocks[1:]
                if start <= block.draw_date <= end
                for row in block.rows
            )
            if values:
                period_means[label] = mean(values)
        if period_means:
            shifts.append(
                TemporalFeatureShift(
                    feature=feature_name,
                    period_means=period_means,
                    maximum_mean_shift=max(period_means.values()) - min(period_means.values()),
                )
            )
    return tuple(sorted(shifts, key=lambda shift: shift.maximum_mean_shift, reverse=True))


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_std = pstdev(left)
    right_std = pstdev(right)
    if left_std == 0 or right_std == 0:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    covariance = mean(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return covariance / (left_std * right_std)


def _standardize(
    rows: list[tuple[float, ...]],
) -> tuple[list[tuple[float, ...]], tuple[float, ...], tuple[float, ...]]:
    columns = tuple(zip(*rows, strict=False))
    means = tuple(mean(column) for column in columns)
    stds = tuple(pstdev(column) if len(column) > 1 else 0.0 for column in columns)
    scaled = [
        tuple(
            0.0 if stds[index] == 0 else (value - means[index]) / stds[index]
            for index, value in enumerate(row)
        )
        for row in rows
    ]
    return scaled, means, stds


def _top_features(
    scores: tuple[tuple[str, float], ...], limit: int = 8
) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(scores, key=lambda item: (-item[1], item[0]))[:limit])


def _load_stage07_comparison(lottery_code: str) -> dict[str, Any]:
    filename = (
        "stage07_loto6_ml_baseline.json"
        if lottery_code == "LOTO6"
        else "stage07_mini_loto_ml_baseline.json"
    )
    path = Path("data") / "exports" / filename
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "loaded",
        "path": str(path),
        "models": {
            name: {
                "mean_matches": model["mean_matches"]["strategy_value"],
                "difference_vs_random": model["mean_matches"]["difference"],
                "conclusion": model["conclusion"],
            }
            for name, model in payload["models"].items()
        },
    }


def _classify_feature_conclusion(ablations: dict[str, AblationResult]) -> str:
    v1 = ablations["v1_all"].best_difference_vs_random
    v2 = ablations["v2_all"].best_difference_vs_random
    best = max(ablations.values(), key=lambda result: result.best_difference_vs_random)
    if v2 <= v1 and best.best_difference_vs_random <= 0:
        return "no_feature_improvement"
    if best.best_difference_vs_random <= 0:
        return "needs_more_validation"
    best_model = best.models[best.best_model]
    if (
        best_model.mean_matches.difference_ci.lower
        <= 0
        <= best_model.mean_matches.difference_ci.upper
    ):
        return "weak_feature_signal"
    positive_periods = sum(
        period.mean_match_difference > 0 for period in best_model.period_stability
    )
    if positive_periods < max(1, len(best_model.period_stability) - 1):
        return "unstable_feature_signal"
    return "promising_for_next_model_stage"
