from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.baseline_benchmark import DEFAULT_STAGE05_SEED
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.ml_baseline import (
    DEFAULT_LOGISTIC_MAX_ITER,
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    DrawFeatureBlock,
    LeakageAudit,
    _derived_seed,
    _make_model,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
    EffectSize,
    bootstrap_confidence_interval,
    holm_adjust_p_values,
    paired_permutation_p_value,
)

STAGE25_SCHEMA_VERSION = "v2-stage25-ranking-discrimination-v1"
STAGE25_DECISION_SCHEMA_VERSION = "v2-stage25-frozen-decision-v1"
STAGE25_DISCOVERY_CUTOFF_DRAW = 1401
STAGE25_HOLDOUT_DRAW = 1402
STAGE25_OUTPUT_DIR = Path("data") / "exports" / "stage25"
STAGE25_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
STAGE25_CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
STAGE25_PRIMARY_ENDPOINTS = ("mean_winner_rank", "top15_capture_rate", "top5_capture_rate")
STAGE25_CHAMPION_C = 1.0
STAGE25_CHAMPION_CALIBRATION = "uncalibrated"
STAGE25_FEATURE_GROUP = "pair_only"
STAGE25_MODEL = "logistic_regression"
STAGE25_PORTFOLIO = "top_ranked"


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    draw_number: int
    draw_date: str
    scores: dict[int, float]
    raw_scores: dict[int, float]
    ranks: dict[int, int]
    winning_numbers: tuple[int, ...]
    winner_ranks: tuple[int, ...]
    random_ranks: dict[int, int]
    random_winner_ranks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScoreCompressionSummary:
    average_min_score: float
    average_max_score: float
    average_mean_score: float
    average_score_stddev: float
    average_score_range: float
    average_iqr: float
    average_top5_cutoff_score: float
    average_top15_cutoff_score: float
    average_margin_rank5_vs_6: float
    average_margin_rank15_vs_16: float
    average_abs_distance_from_half: float
    compression_ratio: float
    winner_mean_score: float
    loser_mean_score: float
    winner_loser_score_difference: float
    standardized_separation: float


@dataclass(frozen=True, slots=True)
class RankDiscriminationSummary:
    mean_winner_rank: float
    median_winner_rank: float
    mean_best_winner_rank: float
    mean_worst_winner_rank: float
    top5_capture_rate: float
    top10_capture_rate: float
    top15_capture_rate: float
    top20_capture_rate: float
    random_mean_winner_rank: float
    random_top5_capture_rate: float
    random_top15_capture_rate: float


@dataclass(frozen=True, slots=True)
class RankStabilitySummary:
    compared_snapshots: int
    average_spearman_rank_correlation: float
    average_top5_jaccard: float
    average_top10_jaccard: float
    average_top15_jaccard: float
    average_top15_entering: float
    average_top15_exiting: float
    average_absolute_rank_movement: float
    average_score_change_magnitude: float


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    method: str
    c_value: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    rank_discrimination: RankDiscriminationSummary


@dataclass(frozen=True, slots=True)
class PrimaryMetricComparison:
    endpoint: str
    challenger_value: float
    champion_value: float
    difference: float
    difference_ci: ConfidenceInterval
    raw_p_value: float
    holm_p_value: float
    bh_p_value: float
    effect_size: EffectSize
    classification: str


@dataclass(frozen=True, slots=True)
class RegularizationResult:
    c_value: float
    calibration: str
    rank_discrimination: RankDiscriminationSummary
    score_compression: ScoreCompressionSummary
    primary_comparisons: dict[str, PrimaryMetricComparison]


@dataclass(frozen=True, slots=True)
class HoldoutObservation:
    evaluated: bool
    draw_number: int | None
    draw_date: str | None
    actual_main_numbers: tuple[int, ...]
    champion_winner_ranks: dict[int, int]
    challenger_winner_ranks: dict[int, int]
    champion_top5_capture: int
    champion_top15_capture: int
    challenger_top5_capture: int
    challenger_top15_capture: int
    note: str


@dataclass(frozen=True, slots=True)
class Stage25RankingDiscriminationResult:
    schema_version: str
    lottery: str
    discovery_cutoff_draw: int
    discovery_dataset_hash: str
    discovery_draw_count: int
    discovery_range: dict[str, str | int]
    champion_configuration: dict[str, Any]
    configuration: dict[str, Any]
    sklearn_version: str
    score_compression: ScoreCompressionSummary
    winner_rank_discrimination: RankDiscriminationSummary
    rank_stability: RankStabilitySummary
    calibration_results: dict[str, CalibrationDiagnostics]
    regularization_results: dict[str, RegularizationResult]
    strongest_challenger: dict[str, Any]
    frozen_decision: dict[str, Any]
    frozen_decision_hash: str
    holdout: HoldoutObservation
    leakage: LeakageAudit
    warnings: tuple[str, ...]


def run_stage25_ranking_discrimination(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    seed: int = DEFAULT_STAGE05_SEED,
    min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    output_dir: str | Path | None = None,
) -> Stage25RankingDiscriminationResult:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 25 ranking discrimination supports MINI_LOTO only")
    if min_training_draws <= 1:
        raise ResearchValidationError("min_training_draws must be greater than 1")
    if refit_interval <= 0:
        raise ResearchValidationError("refit_interval must be positive")
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")

    ordered_input = validate_lottery_dataset(draws, lottery)
    discovery = discovery_slice(ordered_input, lottery)
    if len(discovery) <= min_training_draws:
        raise ResearchValidationError("not enough Mini Loto history for Stage 25")
    discovery_hash = calculate_dataset_hash(discovery)
    feature_names = FEATURE_GROUPS[STAGE25_FEATURE_GROUP]
    blocks = build_walk_forward_feature_blocks(discovery, lottery, feature_names)

    champion = evaluate_score_model(
        discovery,
        lottery,
        blocks,
        seed=seed,
        c_value=STAGE25_CHAMPION_C,
        calibration=STAGE25_CHAMPION_CALIBRATION,
        min_training_draws=min_training_draws,
        refit_interval=refit_interval,
    )
    calibration_results = {
        method: CalibrationDiagnostics(
            method=method,
            c_value=STAGE25_CHAMPION_C,
            brier_score=diagnostics["brier_score"],
            log_loss=diagnostics["log_loss"],
            expected_calibration_error=diagnostics["expected_calibration_error"],
            rank_discrimination=rank_discrimination_summary(diagnostics["snapshots"]),
        )
        for method, diagnostics in (
            (
                method,
                evaluate_score_model(
                    discovery,
                    lottery,
                    blocks,
                    seed=seed,
                    c_value=STAGE25_CHAMPION_C,
                    calibration=method,
                    min_training_draws=min_training_draws,
                    refit_interval=refit_interval,
                ),
            )
            for method in STAGE25_CALIBRATION_METHODS
        )
    }
    candidates = {
        _config_id(c_value, STAGE25_CHAMPION_CALIBRATION): evaluate_score_model(
            discovery,
            lottery,
            blocks,
            seed=seed,
            c_value=c_value,
            calibration=STAGE25_CHAMPION_CALIBRATION,
            min_training_draws=min_training_draws,
            refit_interval=refit_interval,
        )
        for c_value in STAGE25_C_GRID
    }
    candidates.update(
        {
            _config_id(STAGE25_CHAMPION_C, method): evaluate_score_model(
                discovery,
                lottery,
                blocks,
                seed=seed,
                c_value=STAGE25_CHAMPION_C,
                calibration=method,
                min_training_draws=min_training_draws,
                refit_interval=refit_interval,
            )
            for method in STAGE25_CALIBRATION_METHODS
            if method != STAGE25_CHAMPION_CALIBRATION
        }
    )
    comparisons = _all_primary_comparisons(
        champion["snapshots"],
        candidates,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    regularization_results = {
        config_id: RegularizationResult(
            c_value=float(candidate["c_value"]),
            calibration=str(candidate["calibration"]),
            rank_discrimination=rank_discrimination_summary(candidate["snapshots"]),
            score_compression=score_compression_summary(candidate["snapshots"], lottery),
            primary_comparisons=comparisons[config_id],
        )
        for config_id, candidate in candidates.items()
    }
    strongest = strongest_challenger(regularization_results)
    leakage = run_stage25_leakage_audit(
        discovery,
        lottery,
        seed=seed,
        min_training_draws=min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 25 leakage audit failed")

    decision_payload = frozen_decision_payload(
        discovery,
        strongest,
        seed=seed,
        min_training_draws=min_training_draws,
        refit_interval=refit_interval,
        bootstrap_replications=bootstrap_replications,
    )
    decision_hash = stable_payload_hash(decision_payload)
    frozen_decision = {**decision_payload, "decision_hash": decision_hash}
    if output_dir is not None:
        save_stage25_frozen_decision(frozen_decision, output_dir)
    holdout = evaluate_1402_holdout_after_frozen_decision(
        ordered_input,
        lottery,
        frozen_decision=frozen_decision,
        seed=seed,
        min_training_draws=min_training_draws,
        refit_interval=refit_interval,
    )
    result = Stage25RankingDiscriminationResult(
        schema_version=STAGE25_SCHEMA_VERSION,
        lottery=str(lottery.code),
        discovery_cutoff_draw=STAGE25_DISCOVERY_CUTOFF_DRAW,
        discovery_dataset_hash=discovery_hash,
        discovery_draw_count=len(discovery),
        discovery_range={
            "first_draw_number": discovery[0].draw_number,
            "last_draw_number": discovery[-1].draw_number,
            "first_draw_date": discovery[0].draw_date.isoformat(),
            "last_draw_date": discovery[-1].draw_date.isoformat(),
            "excluded_after_cutoff": len(ordered_input) - len(discovery),
        },
        champion_configuration=champion_configuration(seed),
        configuration={
            "seed": seed,
            "feature_group": STAGE25_FEATURE_GROUP,
            "feature_names": feature_names,
            "c_grid": STAGE25_C_GRID,
            "calibration_methods": STAGE25_CALIBRATION_METHODS,
            "primary_endpoints": STAGE25_PRIMARY_ENDPOINTS,
            "min_training_draws": min_training_draws,
            "refit_interval": refit_interval,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "discovery_cutoff_draw": STAGE25_DISCOVERY_CUTOFF_DRAW,
            "holdout_draw": STAGE25_HOLDOUT_DRAW,
        },
        sklearn_version=sklearn.__version__,
        score_compression=score_compression_summary(champion["snapshots"], lottery),
        winner_rank_discrimination=rank_discrimination_summary(champion["snapshots"]),
        rank_stability=rank_stability_summary(champion["snapshots"], lottery),
        calibration_results=calibration_results,
        regularization_results=regularization_results,
        strongest_challenger=strongest,
        frozen_decision=frozen_decision,
        frozen_decision_hash=decision_hash,
        holdout=holdout,
        leakage=leakage,
        warnings=(
            "Stage 25 is historical research only and does not change production strategy.",
            "Discovery is frozen at Mini Loto #1401; #1402 is observational holdout only.",
            "Score calibration can improve probability quality without improving ranking.",
            "Monotonic score transforms do not alter rank-based predictive evidence.",
        ),
    )
    if output_dir is not None:
        save_stage25_outputs(result, output_dir)
    return result


def discovery_slice(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 25 ranking discrimination supports MINI_LOTO only")
    ordered = validate_lottery_dataset(draws, lottery)
    sliced = tuple(draw for draw in ordered if draw.draw_number <= STAGE25_DISCOVERY_CUTOFF_DRAW)
    if not sliced or sliced[-1].draw_number != STAGE25_DISCOVERY_CUTOFF_DRAW:
        raise ResearchValidationError("Stage 25 discovery requires Mini Loto history through #1401")
    return sliced


def champion_configuration(seed: int) -> dict[str, Any]:
    model = _make_logistic_model(seed=seed, c_value=STAGE25_CHAMPION_C)
    return {
        "model": STAGE25_MODEL,
        "feature_group": STAGE25_FEATURE_GROUP,
        "feature_names": FEATURE_GROUPS[STAGE25_FEATURE_GROUP],
        "portfolio_method": STAGE25_PORTFOLIO,
        "calibration": STAGE25_CHAMPION_CALIBRATION,
        "model_parameters": model.get_params(),
    }


def evaluate_score_model(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    blocks: tuple[DrawFeatureBlock, ...],
    *,
    seed: int,
    c_value: float,
    calibration: str,
    min_training_draws: int,
    refit_interval: int,
) -> dict[str, Any]:
    if calibration not in STAGE25_CALIBRATION_METHODS:
        raise ResearchValidationError(f"unsupported calibration method: {calibration}")
    snapshots: list[ScoreSnapshot] = []
    labels: list[int] = []
    probabilities: list[float] = []
    model: LogisticRegression | None = None
    calibrator: LogisticRegression | IsotonicRegression | None = None
    last_fit_index: int | None = None
    for target_index in range(min_training_draws, len(draws)):
        target = draws[target_index]
        if (
            model is None
            or last_fit_index is None
            or target_index - last_fit_index >= refit_interval
        ):
            x_train, y_train, dates = build_training_dataset(blocks, target_index)
            _assert_training_dates(dates, target)
            model = _make_logistic_model(seed=seed, c_value=c_value)
            model.fit(x_train, y_train)
            train_raw = _raw_model_scores(model, x_train)
            calibrator = _fit_calibrator(train_raw, y_train, calibration, seed)
            last_fit_index = target_index
        assert model is not None
        target_rows = blocks[target_index].rows
        raw_scores = _raw_model_scores(model, [row.features for row in target_rows])
        calibrated = _calibrate_scores(raw_scores, calibrator, calibration)
        scores = {row.number: float(calibrated[index]) for index, row in enumerate(target_rows)}
        raw_by_number = {
            row.number: float(raw_scores[index]) for index, row in enumerate(target_rows)
        }
        ranks = rank_numbers_from_scores(scores)
        rng = random.Random(_derived_seed(seed, f"stage25-random-rank-{target.draw_number}"))
        random_order = tuple(
            rng.sample(
                range(lottery.number_min, lottery.number_max + 1),
                lottery.number_max - lottery.number_min + 1,
            )
        )
        random_ranks = {number: index + 1 for index, number in enumerate(random_order)}
        snapshots.append(
            ScoreSnapshot(
                draw_number=target.draw_number,
                draw_date=target.draw_date.isoformat(),
                scores=scores,
                raw_scores=raw_by_number,
                ranks=ranks,
                winning_numbers=target.main_numbers,
                winner_ranks=tuple(ranks[number] for number in target.main_numbers),
                random_ranks=random_ranks,
                random_winner_ranks=tuple(random_ranks[number] for number in target.main_numbers),
            )
        )
        for row, score in zip(target_rows, calibrated, strict=True):
            labels.append(row.label)
            probabilities.append(float(score))
    return {
        "c_value": c_value,
        "calibration": calibration,
        "snapshots": tuple(snapshots),
        "brier_score": _brier_score(labels, probabilities),
        "log_loss": _log_loss(labels, probabilities),
        "expected_calibration_error": _expected_calibration_error(labels, probabilities),
    }


def rank_numbers_from_scores(scores: dict[int, float]) -> dict[int, int]:
    return {
        number: index + 1
        for index, (number, _score) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        )
    }


def monotonic_transform_rankings(scores: dict[int, float]) -> dict[str, dict[int, int]]:
    values = tuple(scores.values())
    avg = mean(values) if values else 0.0
    std = pstdev(values) if len(values) > 1 else 0.0
    ordered = sorted(scores.items(), key=lambda item: item[1])
    percentile_scores = {
        number: index / max(1, len(ordered) - 1) for index, (number, _score) in enumerate(ordered)
    }
    transformed = {
        "probability": scores,
        "decision_function_equivalent": {
            number: math.log(_clip(value) / (1 - _clip(value))) for number, value in scores.items()
        },
        "z_score": {
            number: 0.0 if std == 0 else (value - avg) / std for number, value in scores.items()
        },
        "percentile": percentile_scores,
    }
    return {name: rank_numbers_from_scores(items) for name, items in transformed.items()}


def score_compression_summary(
    snapshots: tuple[ScoreSnapshot, ...],
    lottery: LotteryDefinition,
) -> ScoreCompressionSummary:
    per_draw = tuple(_score_distribution(snapshot) for snapshot in snapshots)
    winner_scores = tuple(
        snapshot.scores[number] for snapshot in snapshots for number in snapshot.winning_numbers
    )
    loser_scores = tuple(
        score
        for snapshot in snapshots
        for number, score in snapshot.scores.items()
        if number not in snapshot.winning_numbers
    )
    all_scores = tuple(score for snapshot in snapshots for score in snapshot.scores.values())
    winner_mean = mean(winner_scores) if winner_scores else 0.0
    loser_mean = mean(loser_scores) if loser_scores else 0.0
    pooled_std = pstdev(all_scores) if len(all_scores) > 1 else 0.0
    return ScoreCompressionSummary(
        average_min_score=_avg(per_draw, "min"),
        average_max_score=_avg(per_draw, "max"),
        average_mean_score=_avg(per_draw, "mean"),
        average_score_stddev=_avg(per_draw, "std"),
        average_score_range=_avg(per_draw, "range"),
        average_iqr=_avg(per_draw, "iqr"),
        average_top5_cutoff_score=_avg(per_draw, "top5_cutoff"),
        average_top15_cutoff_score=_avg(per_draw, "top15_cutoff"),
        average_margin_rank5_vs_6=_avg(per_draw, "margin5"),
        average_margin_rank15_vs_16=_avg(per_draw, "margin15"),
        average_abs_distance_from_half=(
            mean(tuple(abs(score - 0.5) for score in all_scores)) if all_scores else 0.0
        ),
        compression_ratio=_avg(per_draw, "range"),
        winner_mean_score=winner_mean,
        loser_mean_score=loser_mean,
        winner_loser_score_difference=winner_mean - loser_mean,
        standardized_separation=0.0 if pooled_std == 0 else (winner_mean - loser_mean) / pooled_std,
    )


def rank_discrimination_summary(
    snapshots: tuple[ScoreSnapshot, ...],
) -> RankDiscriminationSummary:
    winner_ranks = tuple(rank for snapshot in snapshots for rank in snapshot.winner_ranks)
    random_ranks = tuple(rank for snapshot in snapshots for rank in snapshot.random_winner_ranks)
    return RankDiscriminationSummary(
        mean_winner_rank=mean(winner_ranks) if winner_ranks else 0.0,
        median_winner_rank=median(winner_ranks) if winner_ranks else 0.0,
        mean_best_winner_rank=mean(tuple(min(snapshot.winner_ranks) for snapshot in snapshots))
        if snapshots
        else 0.0,
        mean_worst_winner_rank=mean(tuple(max(snapshot.winner_ranks) for snapshot in snapshots))
        if snapshots
        else 0.0,
        top5_capture_rate=_capture_rate(snapshots, 5),
        top10_capture_rate=_capture_rate(snapshots, 10),
        top15_capture_rate=_capture_rate(snapshots, 15),
        top20_capture_rate=_capture_rate(snapshots, 20),
        random_mean_winner_rank=mean(random_ranks) if random_ranks else 0.0,
        random_top5_capture_rate=_random_capture_rate(snapshots, 5),
        random_top15_capture_rate=_random_capture_rate(snapshots, 15),
    )


def rank_stability_summary(
    snapshots: tuple[ScoreSnapshot, ...],
    lottery: LotteryDefinition,
) -> RankStabilitySummary:
    if len(snapshots) < 2:
        return RankStabilitySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    numbers = tuple(range(lottery.number_min, lottery.number_max + 1))
    spearman: list[float] = []
    top5: list[float] = []
    top10: list[float] = []
    top15: list[float] = []
    entering15: list[int] = []
    exiting15: list[int] = []
    movement: list[float] = []
    score_changes: list[float] = []
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        spearman.append(_spearman(previous.ranks, current.ranks, numbers))
        previous15 = _top_set(previous.ranks, 15)
        current15 = _top_set(current.ranks, 15)
        top5.append(_jaccard(_top_set(previous.ranks, 5), _top_set(current.ranks, 5)))
        top10.append(_jaccard(_top_set(previous.ranks, 10), _top_set(current.ranks, 10)))
        top15.append(_jaccard(previous15, current15))
        entering15.append(len(current15 - previous15))
        exiting15.append(len(previous15 - current15))
        movement.append(
            mean(abs(previous.ranks[number] - current.ranks[number]) for number in numbers)
        )
        score_changes.append(
            mean(abs(previous.scores[number] - current.scores[number]) for number in numbers)
        )
    return RankStabilitySummary(
        compared_snapshots=len(snapshots) - 1,
        average_spearman_rank_correlation=mean(spearman),
        average_top5_jaccard=mean(top5),
        average_top10_jaccard=mean(top10),
        average_top15_jaccard=mean(top15),
        average_top15_entering=mean(entering15),
        average_top15_exiting=mean(exiting15),
        average_absolute_rank_movement=mean(movement),
        average_score_change_magnitude=mean(score_changes),
    )


def strongest_challenger(
    results: dict[str, RegularizationResult],
) -> dict[str, Any]:
    candidates = {
        config_id: result
        for config_id, result in results.items()
        if config_id != _config_id(STAGE25_CHAMPION_C, STAGE25_CHAMPION_CALIBRATION)
    }
    endpoint = "mean_winner_rank"
    config_id, result = max(
        candidates.items(),
        key=lambda item: (
            item[1].primary_comparisons[endpoint].difference,
            -item[1].primary_comparisons[endpoint].holm_p_value,
            item[0],
        ),
    )
    comparisons = result.primary_comparisons
    recommendation = _challenger_recommendation(comparisons)
    return {
        "config_id": config_id,
        "c_value": result.c_value,
        "calibration": result.calibration,
        "primary_endpoint": endpoint,
        "mean_winner_rank": result.rank_discrimination.mean_winner_rank,
        "top5_capture_rate": result.rank_discrimination.top5_capture_rate,
        "top15_capture_rate": result.rank_discrimination.top15_capture_rate,
        "primary_difference": comparisons[endpoint].difference,
        "raw_p_value": comparisons[endpoint].raw_p_value,
        "holm_p_value": comparisons[endpoint].holm_p_value,
        "bh_p_value": comparisons[endpoint].bh_p_value,
        "classification": comparisons[endpoint].classification,
        "recommendation": recommendation,
    }


def frozen_decision_payload(
    discovery: tuple[HistoricalDraw, ...],
    strongest: dict[str, Any],
    *,
    seed: int,
    min_training_draws: int,
    refit_interval: int,
    bootstrap_replications: int,
) -> dict[str, Any]:
    return {
        "schema_version": STAGE25_DECISION_SCHEMA_VERSION,
        "lottery": "MINI_LOTO",
        "discovery_cutoff_draw": STAGE25_DISCOVERY_CUTOFF_DRAW,
        "discovery_dataset_hash": calculate_dataset_hash(discovery),
        "discovery_draw_count": len(discovery),
        "discovery_last_draw_date": discovery[-1].draw_date.isoformat(),
        "champion_configuration": champion_configuration(seed),
        "tested_c_grid": STAGE25_C_GRID,
        "tested_calibration_methods": STAGE25_CALIBRATION_METHODS,
        "primary_endpoints": STAGE25_PRIMARY_ENDPOINTS,
        "strongest_challenger": strongest,
        "challenger_recommendation": strongest["recommendation"],
        "configuration": {
            "seed": seed,
            "min_training_draws": min_training_draws,
            "refit_interval": refit_interval,
            "bootstrap_replications": bootstrap_replications,
            "multiplicity_correction": "Holm across Stage 25 challenger endpoint tests",
            "bh_exploratory_reported": True,
        },
        "frozen_before_holdout": True,
        "excluded_from_discovery": {
            "holdout_draw": STAGE25_HOLDOUT_DRAW,
            "later_draws": "all draws after #1402",
        },
    }


def evaluate_1402_holdout_after_frozen_decision(
    all_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    frozen_decision: dict[str, Any],
    seed: int,
    min_training_draws: int,
    refit_interval: int,
) -> HoldoutObservation:
    if not frozen_decision.get("frozen_before_holdout"):
        raise ResearchValidationError("Stage 25 decision must be frozen before holdout evaluation")
    discovery = discovery_slice(all_draws, lottery)
    holdout = next((draw for draw in all_draws if draw.draw_number == STAGE25_HOLDOUT_DRAW), None)
    if holdout is None:
        return HoldoutObservation(
            False,
            None,
            None,
            (),
            {},
            {},
            0,
            0,
            0,
            0,
            "Mini Loto #1402 is not available in the supplied data.",
        )
    feature_names = FEATURE_GROUPS[STAGE25_FEATURE_GROUP]
    draws_for_holdout = (*discovery, holdout)
    blocks = build_walk_forward_feature_blocks(draws_for_holdout, lottery, feature_names)
    champion = evaluate_score_model(
        draws_for_holdout,
        lottery,
        blocks,
        seed=seed,
        c_value=STAGE25_CHAMPION_C,
        calibration=STAGE25_CHAMPION_CALIBRATION,
        min_training_draws=len(discovery),
        refit_interval=refit_interval,
    )["snapshots"][0]
    challenger_config = frozen_decision["strongest_challenger"]
    challenger = evaluate_score_model(
        draws_for_holdout,
        lottery,
        blocks,
        seed=seed,
        c_value=float(challenger_config["c_value"]),
        calibration=str(challenger_config["calibration"]),
        min_training_draws=len(discovery),
        refit_interval=refit_interval,
    )["snapshots"][0]
    return HoldoutObservation(
        evaluated=True,
        draw_number=holdout.draw_number,
        draw_date=holdout.draw_date.isoformat(),
        actual_main_numbers=holdout.main_numbers,
        champion_winner_ranks={number: champion.ranks[number] for number in holdout.main_numbers},
        challenger_winner_ranks={
            number: challenger.ranks[number] for number in holdout.main_numbers
        },
        champion_top5_capture=sum(champion.ranks[number] <= 5 for number in holdout.main_numbers),
        champion_top15_capture=sum(champion.ranks[number] <= 15 for number in holdout.main_numbers),
        challenger_top5_capture=sum(
            challenger.ranks[number] <= 5 for number in holdout.main_numbers
        ),
        challenger_top15_capture=sum(
            challenger.ranks[number] <= 15 for number in holdout.main_numbers
        ),
        note="One draw is observational only and does not alter the frozen Stage 25 decision.",
    )


def run_stage25_leakage_audit(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(min_training_draws, 2), len(draws) - 2)
    feature_names = FEATURE_GROUPS[STAGE25_FEATURE_GROUP]
    original_blocks = build_walk_forward_feature_blocks(draws, lottery, feature_names)
    target = draws[target_index]
    original = evaluate_score_model(
        draws[: target_index + 1],
        lottery,
        original_blocks[: target_index + 1],
        seed=seed,
        c_value=STAGE25_CHAMPION_C,
        calibration=STAGE25_CHAMPION_CALIBRATION,
        min_training_draws=target_index,
        refit_interval=1,
    )["snapshots"][0]
    target_mutation = list(draws)
    target_mutation[target_index] = HistoricalDraw(
        target.lottery,
        target.draw_number,
        target.draw_date,
        tuple(reversed(target.main_numbers)),
        target.bonus_numbers,
    )
    target_blocks = build_walk_forward_feature_blocks(
        tuple(target_mutation), lottery, feature_names
    )
    mutated_target = evaluate_score_model(
        tuple(target_mutation[: target_index + 1]),
        lottery,
        target_blocks[: target_index + 1],
        seed=seed,
        c_value=STAGE25_CHAMPION_C,
        calibration=STAGE25_CHAMPION_CALIBRATION,
        min_training_draws=target_index,
        refit_interval=1,
    )["snapshots"][0]
    future_mutation = list(draws)
    future = future_mutation[target_index + 1]
    future_mutation[target_index + 1] = HistoricalDraw(
        future.lottery,
        future.draw_number,
        future.draw_date,
        tuple(reversed(future.main_numbers)),
        future.bonus_numbers,
    )
    future_blocks = build_walk_forward_feature_blocks(
        tuple(future_mutation), lottery, feature_names
    )
    mutated_future = evaluate_score_model(
        tuple(future_mutation[: target_index + 2]),
        lottery,
        future_blocks[: target_index + 2],
        seed=seed,
        c_value=STAGE25_CHAMPION_C,
        calibration=STAGE25_CHAMPION_CALIBRATION,
        min_training_draws=target_index,
        refit_interval=1,
    )["snapshots"][0]
    training_dates = build_training_dataset(original_blocks, target_index)[2]
    training_ok = all(
        training_date < target.draw_date.isoformat() for training_date in training_dates
    )
    target_changed = original.scores != mutated_target.scores
    future_changed = original.scores != mutated_future.scores
    return LeakageAudit(
        lookahead_safe=training_ok and not target_changed and not future_changed,
        training_dates_strictly_before_target=training_ok,
        target_mutation_changes_features=target_changed,
        future_mutation_changes_prediction=future_changed,
    )


def stable_payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(research_result_json(payload).encode("utf-8")).hexdigest()


def save_stage25_outputs(
    result: Stage25RankingDiscriminationResult,
    output_dir: str | Path = STAGE25_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    decision_path = save_stage25_frozen_decision(result.frozen_decision, root)
    result_path = root / "v2_stage25_ranking_discrimination.json"
    summary_path = root / "v2_stage25_summary.json"
    result_path.write_text(research_result_json(result), encoding="utf-8")
    summary_path.write_text(
        research_result_json(
            {
                "schema_version": "v2-stage25-summary-v1",
                "lottery": result.lottery,
                "discovery_cutoff_draw": result.discovery_cutoff_draw,
                "discovery_dataset_hash": result.discovery_dataset_hash,
                "mean_winner_rank": result.winner_rank_discrimination.mean_winner_rank,
                "top5_capture_rate": result.winner_rank_discrimination.top5_capture_rate,
                "top15_capture_rate": result.winner_rank_discrimination.top15_capture_rate,
                "strongest_challenger": result.strongest_challenger,
                "frozen_decision_hash": result.frozen_decision_hash,
                "holdout": result.holdout,
            }
        ),
        encoding="utf-8",
    )
    return {
        "decision": str(decision_path),
        "result": str(result_path),
        "summary": str(summary_path),
    }


def save_stage25_frozen_decision(
    frozen_decision: dict[str, Any],
    output_dir: str | Path = STAGE25_OUTPUT_DIR,
) -> Path:
    path = Path(output_dir) / "v2_stage25_frozen_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(frozen_decision), encoding="utf-8")
    return path


def _all_primary_comparisons(
    champion: tuple[ScoreSnapshot, ...],
    candidates: dict[str, dict[str, Any]],
    *,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
) -> dict[str, dict[str, PrimaryMetricComparison]]:
    raw_values: dict[str, float] = {}
    diff_values: dict[str, tuple[float, ...]] = {}
    baseline_values: dict[str, tuple[float, ...]] = {}
    candidate_values: dict[str, tuple[float, ...]] = {}
    for config_id, candidate in candidates.items():
        if config_id == _config_id(STAGE25_CHAMPION_C, STAGE25_CHAMPION_CALIBRATION):
            continue
        for endpoint in STAGE25_PRIMARY_ENDPOINTS:
            label = f"{config_id}:{endpoint}"
            differences, champion_series, candidate_series = _endpoint_differences(
                champion, candidate["snapshots"], endpoint
            )
            diff_values[label] = differences
            baseline_values[label] = champion_series
            candidate_values[label] = candidate_series
            raw_values[label] = paired_permutation_p_value(
                differences,
                seed=_derived_seed(seed, f"stage25-{label}-permutation"),
                replications=bootstrap_replications,
            )
    holm = holm_adjust_p_values(raw_values)
    bh = benjamini_hochberg_adjust_p_values(raw_values)
    results: dict[str, dict[str, PrimaryMetricComparison]] = {
        config_id: {} for config_id in candidates
    }
    for config_id, candidate in candidates.items():
        if config_id == _config_id(STAGE25_CHAMPION_C, STAGE25_CHAMPION_CALIBRATION):
            baseline = rank_discrimination_summary(candidate["snapshots"])
            results[config_id] = {
                endpoint: _baseline_primary_comparison(endpoint, baseline)
                for endpoint in STAGE25_PRIMARY_ENDPOINTS
            }
            continue
        for endpoint in STAGE25_PRIMARY_ENDPOINTS:
            label = f"{config_id}:{endpoint}"
            difference = mean(diff_values[label]) if diff_values[label] else 0.0
            effect = _effect_size(
                diff_values[label],
                mean(baseline_values[label]) if baseline_values[label] else 0.0,
            )
            ci = bootstrap_confidence_interval(
                diff_values[label],
                seed=_derived_seed(seed, f"stage25-{label}-ci"),
                replications=bootstrap_replications,
                confidence_level=confidence_level,
            )
            results[config_id][endpoint] = PrimaryMetricComparison(
                endpoint=endpoint,
                challenger_value=mean(candidate_values[label]) if candidate_values[label] else 0.0,
                champion_value=mean(baseline_values[label]) if baseline_values[label] else 0.0,
                difference=difference,
                difference_ci=ci,
                raw_p_value=raw_values[label],
                holm_p_value=holm[label],
                bh_p_value=bh[label],
                effect_size=effect,
                classification=_classification(difference, ci, holm[label]),
            )
    return results


def _endpoint_differences(
    champion: tuple[ScoreSnapshot, ...],
    challenger: tuple[ScoreSnapshot, ...],
    endpoint: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    champion_values = tuple(_endpoint_value(snapshot, endpoint) for snapshot in champion)
    challenger_values = tuple(_endpoint_value(snapshot, endpoint) for snapshot in challenger)
    if endpoint == "mean_winner_rank":
        differences = tuple(
            left - right for left, right in zip(champion_values, challenger_values, strict=True)
        )
    else:
        differences = tuple(
            right - left for left, right in zip(champion_values, challenger_values, strict=True)
        )
    return differences, champion_values, challenger_values


def _endpoint_value(snapshot: ScoreSnapshot, endpoint: str) -> float:
    if endpoint == "mean_winner_rank":
        return mean(snapshot.winner_ranks)
    if endpoint == "top15_capture_rate":
        return sum(rank <= 15 for rank in snapshot.winner_ranks) / len(snapshot.winner_ranks)
    if endpoint == "top5_capture_rate":
        return sum(rank <= 5 for rank in snapshot.winner_ranks) / len(snapshot.winner_ranks)
    raise ResearchValidationError(f"unknown Stage 25 endpoint: {endpoint}")


def _baseline_primary_comparison(
    endpoint: str,
    summary: RankDiscriminationSummary,
) -> PrimaryMetricComparison:
    value = getattr(summary, endpoint)
    ci = ConfidenceInterval(DEFAULT_CONFIDENCE_LEVEL, value, value)
    effect = EffectSize(0.0, 0.0, 0.0)
    return PrimaryMetricComparison(
        endpoint, value, value, 0.0, ci, 1.0, 1.0, 1.0, effect, "BASELINE"
    )


def _make_logistic_model(seed: int, c_value: float) -> LogisticRegression:
    model = _make_model(STAGE25_MODEL, seed)
    model.set_params(C=c_value, max_iter=DEFAULT_LOGISTIC_MAX_ITER)
    return model


def _raw_model_scores(
    model: LogisticRegression, rows: list[tuple[float, ...]]
) -> tuple[float, ...]:
    return tuple(float(value) for value in model.decision_function(rows))


def _fit_calibrator(
    raw_scores: tuple[float, ...],
    labels: list[int],
    calibration: str,
    seed: int,
) -> LogisticRegression | IsotonicRegression | None:
    if calibration == "uncalibrated":
        return None
    x_values = [(score,) for score in raw_scores]
    if calibration == "sigmoid":
        calibrator = LogisticRegression(
            class_weight=None,
            max_iter=DEFAULT_LOGISTIC_MAX_ITER,
            random_state=seed,
            solver="liblinear",
        )
        calibrator.fit(x_values, labels)
        return calibrator
    if calibration == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_scores, labels)
        return calibrator
    raise ResearchValidationError(f"unsupported calibration method: {calibration}")


def _calibrate_scores(
    raw_scores: tuple[float, ...],
    calibrator: LogisticRegression | IsotonicRegression | None,
    calibration: str,
) -> tuple[float, ...]:
    if calibration == "uncalibrated":
        return tuple(_sigmoid(score) for score in raw_scores)
    if isinstance(calibrator, LogisticRegression):
        probabilities = calibrator.predict_proba([(score,) for score in raw_scores])
        class_index = list(calibrator.classes_).index(1)
        return tuple(float(row[class_index]) for row in probabilities)
    if isinstance(calibrator, IsotonicRegression):
        return tuple(float(score) for score in calibrator.predict(raw_scores))
    raise ResearchValidationError(f"calibrator missing for {calibration}")


def _score_distribution(snapshot: ScoreSnapshot) -> dict[str, float]:
    values = tuple(sorted(snapshot.scores.values(), reverse=True))
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "range": max(values) - min(values),
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
        "top5_cutoff": values[4],
        "top15_cutoff": values[14],
        "margin5": values[4] - values[5],
        "margin15": values[14] - values[15],
    }


def _avg(rows: tuple[dict[str, float], ...], key: str) -> float:
    return mean(tuple(row[key] for row in rows)) if rows else 0.0


def _capture_rate(snapshots: tuple[ScoreSnapshot, ...], rank_cutoff: int) -> float:
    denominator = sum(len(snapshot.winner_ranks) for snapshot in snapshots)
    if denominator == 0:
        return 0.0
    return (
        sum(rank <= rank_cutoff for snapshot in snapshots for rank in snapshot.winner_ranks)
        / denominator
    )


def _random_capture_rate(snapshots: tuple[ScoreSnapshot, ...], rank_cutoff: int) -> float:
    denominator = sum(len(snapshot.random_winner_ranks) for snapshot in snapshots)
    if denominator == 0:
        return 0.0
    return (
        sum(rank <= rank_cutoff for snapshot in snapshots for rank in snapshot.random_winner_ranks)
        / denominator
    )


def _top_set(ranks: dict[int, int], top_k: int) -> set[int]:
    return {number for number, rank in ranks.items() if rank <= top_k}


def _jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _spearman(left: dict[int, int], right: dict[int, int], numbers: tuple[int, ...]) -> float:
    n = len(numbers)
    if n < 2:
        return 0.0
    squared = sum((left[number] - right[number]) ** 2 for number in numbers)
    return 1 - (6 * squared) / (n * (n * n - 1))


def _brier_score(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    return mean(
        (_clip(probability) - label) ** 2
        for label, probability in zip(labels, probabilities, strict=True)
    )


def _log_loss(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    return -mean(
        label * math.log(_clip(probability)) + (1 - label) * math.log(1 - _clip(probability))
        for label, probability in zip(labels, probabilities, strict=True)
    )


def _expected_calibration_error(
    labels: list[int],
    probabilities: list[float],
    bins: int = 10,
) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = tuple(
            (label, probability)
            for label, probability in zip(labels, probabilities, strict=True)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        )
        if selected:
            avg_confidence = mean(tuple(probability for _label, probability in selected))
            avg_label = mean(tuple(label for label, _probability in selected))
            error += (len(selected) / total) * abs(avg_confidence - avg_label)
    return error


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    std = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=0.0 if std == 0 else absolute / std,
    )


def _classification(
    difference: float,
    ci: ConfidenceInterval,
    adjusted_p_value: float,
) -> str:
    if ci.upper < 0 and adjusted_p_value < 0.05:
        return "NEGATIVE"
    if difference < 0 and ci.upper <= 0:
        return "NEGATIVE"
    if ci.lower > 0 and adjusted_p_value < 0.05:
        return "EVIDENCE"
    if difference > 0 and adjusted_p_value < 0.1:
        return "WEAK_SIGNAL"
    if adjusted_p_value >= 0.1 and ci.lower <= 0 <= ci.upper:
        return "NO_EVIDENCE"
    return "INCONCLUSIVE"


def _challenger_recommendation(comparisons: dict[str, PrimaryMetricComparison]) -> str:
    primary = comparisons["mean_winner_rank"]
    top15 = comparisons["top15_capture_rate"]
    top5 = comparisons["top5_capture_rate"]
    if (
        primary.difference > 0
        and top15.difference >= 0
        and top5.difference >= 0
        and primary.classification in {"EVIDENCE", "WEAK_SIGNAL"}
        and primary.difference_ci.upper >= 0
    ):
        return "KEEP_AS_SHADOW_CANDIDATE"
    return "NONE"


def _assert_training_dates(dates: tuple[str, ...], target: HistoricalDraw) -> None:
    if not all(date_text < target.draw_date.isoformat() for date_text in dates):
        raise ResearchValidationError("training rows include target or future draw")


def _config_id(c_value: float, calibration: str) -> str:
    return f"c_{str(c_value).replace('.', '_')}_{calibration}"


def _quantile(values: tuple[float, ...], q: float) -> float:
    if not values:
        return 0.0
    ordered = tuple(sorted(values))
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _clip(value: float, eps: float = 1e-12) -> float:
    return min(max(value, eps), 1.0 - eps)
