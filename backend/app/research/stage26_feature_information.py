from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.baseline_benchmark import DEFAULT_STAGE05_SEED
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    FEATURE_NAMES_V2,
    LeakageAudit,
    _derived_seed,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.stage25_ranking_discrimination import (
    evaluate_score_model,
    rank_numbers_from_scores,
)
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
    EffectSize,
    holm_adjust_p_values,
)

STAGE26_SCHEMA_VERSION = "v2-stage26-feature-information-v1"
STAGE26_DECISION_SCHEMA_VERSION = "v2-stage26-frozen-decision-v1"
STAGE26_DISCOVERY_CUTOFF_DRAW = 1401
STAGE26_HOLDOUT_DRAW = 1402
STAGE26_OUTPUT_DIR = Path("data") / "exports" / "stage26"
STAGE26_ROLLING_WINDOW = 100
STAGE26_PRIMARY_ENDPOINTS = ("mean_winner_rank", "top5_capture_rate", "top15_capture_rate")


FEATURE_DIRECTIONS: dict[str, str] = {
    "frequency_rate": "higher",
    "frequency_5": "higher",
    "frequency_10": "higher",
    "frequency_20": "higher",
    "frequency_50": "higher",
    "frequency_100": "higher",
    "current_gap": "lower",
    "mean_gap": "lower",
    "gap_std": "lower",
    "max_gap": "lower",
    "recent_activity_10": "higher",
    "pair_strength_rate": "higher",
    "frequency_momentum_5_20": "higher",
    "frequency_momentum_10_50": "higher",
    "frequency_ratio_5_20": "higher",
    "gap_to_mean": "lower",
    "gap_to_median": "lower",
    "gap_z_score": "lower",
    "seen_rate_per_draw": "higher",
    "pair_mean_strength_rate": "higher",
    "pair_max_strength_rate": "higher",
    "previous_draw_presence": "higher",
    "previous_draw_pair_strength_rate": "higher",
}

FEATURE_DEFINITIONS: dict[str, str] = {
    "frequency_rate": "total appearances divided by total historical main-number slots",
    "frequency_5": "appearances in the previous 5 draws",
    "frequency_10": "appearances in the previous 10 draws",
    "frequency_20": "appearances in the previous 20 draws",
    "frequency_50": "appearances in the previous 50 draws",
    "frequency_100": "appearances in the previous 100 draws",
    "current_gap": "draws since the number was last seen in main numbers",
    "mean_gap": "mean historical gap between main-number appearances",
    "gap_std": "population standard deviation of historical appearance gaps",
    "max_gap": "maximum historical gap between main-number appearances",
    "recent_activity_10": "same historical quantity as frequency_10",
    "pair_strength_rate": "sum of historical pair co-occurrences involving the number per draw",
    "frequency_momentum_5_20": "previous 5-draw rate minus previous 20-draw rate",
    "frequency_momentum_10_50": "previous 10-draw rate minus previous 50-draw rate",
    "frequency_ratio_5_20": "previous 5-draw rate divided by previous 20-draw rate",
    "gap_to_mean": "current_gap divided by mean_gap",
    "gap_to_median": "current_gap divided by median historical gap",
    "gap_z_score": "current_gap standardized by historical gap mean and standard deviation",
    "seen_rate_per_draw": "main-number appearances divided by historical draw count",
    "pair_mean_strength_rate": "mean incident pair count divided by historical draw count",
    "pair_max_strength_rate": "maximum incident pair count divided by historical draw count",
    "previous_draw_presence": "1 if the number appeared in the immediately previous draw",
    "previous_draw_pair_strength_rate": (
        "historical pair strength between the number and previous draw numbers per draw"
    ),
}


@dataclass(frozen=True, slots=True)
class FeatureInventoryItem:
    name: str
    feature_group: str
    definition: str
    expected_direction: str
    required_lookback: str
    target_draw_included: bool


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    draw_number: int
    draw_date: str
    values: dict[int, float]
    ranks: dict[int, int]
    inverse_ranks: dict[int, int]
    random_ranks: dict[int, int]
    winning_numbers: tuple[int, ...]
    winner_values: tuple[float, ...]
    loser_values: tuple[float, ...]
    winner_ranks: tuple[int, ...]
    inverse_winner_ranks: tuple[int, ...]
    random_winner_ranks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SeparationResult:
    winner_mean: float
    loser_mean: float
    winner_minus_loser: float
    direction_adjusted_difference: float
    effect_size: EffectSize
    confidence_interval: ConfidenceInterval
    raw_p_value: float
    holm_p_value: float
    bh_p_value: float
    classification: str


@dataclass(frozen=True, slots=True)
class RankEndpointResult:
    endpoint: str
    feature_value: float
    random_value: float
    difference: float
    confidence_interval: ConfidenceInterval
    raw_p_value: float
    holm_p_value: float
    bh_p_value: float
    effect_size: EffectSize
    classification: str


@dataclass(frozen=True, slots=True)
class RankerResult:
    sample_size: int
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
    primary_endpoints: dict[str, RankEndpointResult]


@dataclass(frozen=True, slots=True)
class PeriodStabilityResult:
    period: str
    target_draws: int
    mean_winner_rank: float
    random_mean_winner_rank: float
    mean_rank_advantage: float
    top5_capture_rate: float
    random_top5_capture_rate: float
    top15_capture_rate: float
    random_top15_capture_rate: float


@dataclass(frozen=True, slots=True)
class RollingStabilityResult:
    window_size: int
    window_count: int
    positive_mean_rank_advantage_fraction: float
    positive_top15_advantage_fraction: float
    median_mean_rank_advantage: float
    minimum_mean_rank_advantage: float
    maximum_mean_rank_advantage: float
    recent_mean_rank_advantage: float
    recent_top15_advantage: float
    longest_positive_mean_rank_streak: int


@dataclass(frozen=True, slots=True)
class FeatureInformationResult:
    name: str
    inventory: FeatureInventoryItem
    separation: SeparationResult
    direct_ranker: RankerResult
    inverse_ranker: RankerResult
    inverse_is_diagnostic: bool
    period_stability: tuple[PeriodStabilityResult, ...]
    stability_classification: str
    rolling_stability: RollingStabilityResult
    recommendation: str


@dataclass(frozen=True, slots=True)
class RedundancyResult:
    average_pairwise_spearman: float
    high_redundancy_pairs: tuple[tuple[str, str, float], ...]
    relatively_independent_pairs: tuple[tuple[str, str, float], ...]
    pair_strength_redundancy: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ChampionAttributionResult:
    direct_feature: str
    lr_feature_group: str
    rank_order_equality_rate: float
    average_spearman: float
    top5_membership_equality_rate: float
    top15_membership_equality_rate: float
    winner_rank_difference_mean: float
    top5_capture_difference: float
    top15_capture_difference: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    evaluated: bool
    draw_number: int | None
    draw_date: str | None
    feature: str
    selected_top5: tuple[int, ...]
    selected_top15: tuple[int, ...]
    actual_main_numbers: tuple[int, ...]
    actual_winner_ranks: dict[int, int]
    top5_winners_captured: int
    top15_winners_captured: int


@dataclass(frozen=True, slots=True)
class Stage26FeatureInformationAudit:
    schema_version: str
    lottery: str
    discovery_cutoff_draw: int
    discovery_dataset_hash: str
    discovery_draw_count: int
    discovery_range: dict[str, str | int]
    configuration: dict[str, Any]
    feature_inventory: dict[str, FeatureInventoryItem]
    features: dict[str, FeatureInformationResult]
    redundancy: RedundancyResult
    champion_attribution: ChampionAttributionResult
    leakage: LeakageAudit
    strongest_feature: dict[str, Any]
    frozen_decision: dict[str, Any]
    frozen_decision_hash: str
    holdout: HoldoutResult
    stage27_feature_recommendation: str
    stage27_ensemble_recommendation: str
    warnings: tuple[str, ...]


def run_stage26_feature_information_audit(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    seed: int = DEFAULT_STAGE05_SEED,
    min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    output_dir: str | Path | None = None,
) -> Stage26FeatureInformationAudit:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 26 feature information audit supports MINI_LOTO only")
    ordered_input = validate_lottery_dataset(draws, lottery)
    discovery = discovery_slice(ordered_input, lottery)
    if len(discovery) <= min_training_draws:
        raise ResearchValidationError("not enough Mini Loto history for Stage 26")
    inventory = feature_inventory()
    blocks = build_walk_forward_feature_blocks(discovery, lottery, FEATURE_NAMES_V2)
    snapshots = build_feature_snapshots(
        blocks,
        discovery,
        lottery,
        seed=seed,
        min_training_draws=min_training_draws,
    )
    feature_results = evaluate_features(
        snapshots,
        inventory,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    redundancy = redundancy_analysis(snapshots, tuple(inventory))
    champion = champion_attribution(
        discovery,
        lottery,
        feature_snapshots=snapshots["pair_strength_rate"],
        seed=seed,
        min_training_draws=min_training_draws,
    )
    leakage = run_stage26_leakage_audit(
        discovery,
        lottery,
        seed=seed,
        min_training_draws=min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 26 leakage audit failed")
    strongest = strongest_feature(feature_results)
    feature_recommendation = stage27_feature_recommendation(feature_results, redundancy)
    ensemble_recommendation = stage27_ensemble_recommendation(feature_results, redundancy)
    decision_payload = frozen_decision_payload(
        discovery,
        inventory,
        strongest,
        feature_recommendation,
        ensemble_recommendation,
        redundancy,
        seed=seed,
        min_training_draws=min_training_draws,
        bootstrap_replications=bootstrap_replications,
    )
    decision_hash = stable_payload_hash(decision_payload)
    frozen_decision = {**decision_payload, "decision_hash": decision_hash}
    if output_dir is not None:
        save_stage26_frozen_decision(frozen_decision, output_dir)
    holdout = evaluate_holdout_after_frozen_decision(
        ordered_input,
        lottery,
        frozen_decision=frozen_decision,
        seed=seed,
    )
    result = Stage26FeatureInformationAudit(
        schema_version=STAGE26_SCHEMA_VERSION,
        lottery=str(lottery.code),
        discovery_cutoff_draw=STAGE26_DISCOVERY_CUTOFF_DRAW,
        discovery_dataset_hash=calculate_dataset_hash(discovery),
        discovery_draw_count=len(discovery),
        discovery_range={
            "first_draw_number": discovery[0].draw_number,
            "last_draw_number": discovery[-1].draw_number,
            "first_draw_date": discovery[0].draw_date.isoformat(),
            "last_draw_date": discovery[-1].draw_date.isoformat(),
            "excluded_after_cutoff": len(ordered_input) - len(discovery),
        },
        configuration={
            "seed": seed,
            "min_training_draws": min_training_draws,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "primary_endpoints": STAGE26_PRIMARY_ENDPOINTS,
            "rolling_window": STAGE26_ROLLING_WINDOW,
            "feature_names": FEATURE_NAMES_V2,
            "multiplicity": "Holm across feature x primary endpoint hypotheses",
        },
        feature_inventory=inventory,
        features=feature_results,
        redundancy=redundancy,
        champion_attribution=champion,
        leakage=leakage,
        strongest_feature=strongest,
        frozen_decision=frozen_decision,
        frozen_decision_hash=decision_hash,
        holdout=holdout,
        stage27_feature_recommendation=feature_recommendation,
        stage27_ensemble_recommendation=ensemble_recommendation,
        warnings=(
            "Stage 26 is research-only and does not alter production predictions or settings.",
            "Discovery is frozen at Mini Loto #1401; #1402 is observational holdout only.",
            "Inverse-direction rankings are diagnostic only and are not confirmatory evidence.",
            "Historical feature behavior does not guarantee future lottery outcomes.",
        ),
    )
    if output_dir is not None:
        save_stage26_outputs(result, output_dir)
    return result


def discovery_slice(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 26 feature information audit supports MINI_LOTO only")
    ordered = validate_lottery_dataset(draws, lottery)
    sliced = tuple(draw for draw in ordered if draw.draw_number <= STAGE26_DISCOVERY_CUTOFF_DRAW)
    if not sliced or sliced[-1].draw_number != STAGE26_DISCOVERY_CUTOFF_DRAW:
        raise ResearchValidationError("Stage 26 discovery requires Mini Loto history through #1401")
    return sliced


def feature_inventory() -> dict[str, FeatureInventoryItem]:
    groups_by_feature: dict[str, set[str]] = {name: set() for name in FEATURE_NAMES_V2}
    for group_name, names in FEATURE_GROUPS.items():
        for name in names:
            if name in groups_by_feature:
                groups_by_feature[name].add(group_name)
    return {
        name: FeatureInventoryItem(
            name=name,
            feature_group=",".join(sorted(groups_by_feature[name])) or "number-features-v2",
            definition=FEATURE_DEFINITIONS[name],
            expected_direction=FEATURE_DIRECTIONS[name],
            required_lookback=_required_lookback(name),
            target_draw_included=False,
        )
        for name in FEATURE_NAMES_V2
    }


def build_feature_snapshots(
    blocks: tuple[Any, ...],
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    min_training_draws: int,
) -> dict[str, tuple[FeatureSnapshot, ...]]:
    rows: dict[str, list[FeatureSnapshot]] = {name: [] for name in FEATURE_NAMES_V2}
    for target_index in range(min_training_draws, len(draws)):
        block = blocks[target_index]
        target = draws[target_index]
        for feature_index, feature_name in enumerate(FEATURE_NAMES_V2):
            values = {row.number: float(row.features[feature_index]) for row in block.rows}
            scores = _directional_scores(values, FEATURE_DIRECTIONS[feature_name])
            inverse_scores = {number: -score for number, score in scores.items()}
            ranks = rank_numbers_from_scores(scores)
            inverse_ranks = rank_numbers_from_scores(inverse_scores)
            random_ranks = _random_ranks(lottery, seed, feature_name, target.draw_number)
            winners = set(target.main_numbers)
            rows[feature_name].append(
                FeatureSnapshot(
                    draw_number=target.draw_number,
                    draw_date=target.draw_date.isoformat(),
                    values=values,
                    ranks=ranks,
                    inverse_ranks=inverse_ranks,
                    random_ranks=random_ranks,
                    winning_numbers=target.main_numbers,
                    winner_values=tuple(values[number] for number in target.main_numbers),
                    loser_values=tuple(
                        value for number, value in values.items() if number not in winners
                    ),
                    winner_ranks=tuple(ranks[number] for number in target.main_numbers),
                    inverse_winner_ranks=tuple(
                        inverse_ranks[number] for number in target.main_numbers
                    ),
                    random_winner_ranks=tuple(
                        random_ranks[number] for number in target.main_numbers
                    ),
                )
            )
    return {name: tuple(items) for name, items in rows.items()}


def evaluate_features(
    snapshots: dict[str, tuple[FeatureSnapshot, ...]],
    inventory: dict[str, FeatureInventoryItem],
    *,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
) -> dict[str, FeatureInformationResult]:
    separation_diffs = {
        name: _separation_differences(items, inventory[name].expected_direction)
        for name, items in snapshots.items()
    }
    endpoint_diffs: dict[str, tuple[float, ...]] = {}
    for name, items in snapshots.items():
        for endpoint in STAGE26_PRIMARY_ENDPOINTS:
            endpoint_diffs[f"{name}:{endpoint}"] = _endpoint_differences(items, endpoint)
    separation_raw = {name: _paired_mean_p_value(diffs) for name, diffs in separation_diffs.items()}
    endpoint_raw = {key: _paired_mean_p_value(diffs) for key, diffs in endpoint_diffs.items()}
    separation_holm = holm_adjust_p_values(separation_raw)
    separation_bh = benjamini_hochberg_adjust_p_values(separation_raw)
    endpoint_holm = holm_adjust_p_values(endpoint_raw)
    endpoint_bh = benjamini_hochberg_adjust_p_values(endpoint_raw)
    results: dict[str, FeatureInformationResult] = {}
    for name, items in snapshots.items():
        separation = _separation_result(
            items,
            inventory[name].expected_direction,
            separation_diffs[name],
            raw_p=separation_raw[name],
            holm_p=separation_holm[name],
            bh_p=separation_bh[name],
            seed=seed,
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
        )
        endpoints = {
            endpoint: _rank_endpoint_result(
                items,
                endpoint,
                endpoint_diffs[f"{name}:{endpoint}"],
                raw_p=endpoint_raw[f"{name}:{endpoint}"],
                holm_p=endpoint_holm[f"{name}:{endpoint}"],
                bh_p=endpoint_bh[f"{name}:{endpoint}"],
                seed=seed,
                bootstrap_replications=bootstrap_replications,
                confidence_level=confidence_level,
            )
            for endpoint in STAGE26_PRIMARY_ENDPOINTS
        }
        direct = ranker_result(items, endpoints, inverse=False)
        inverse = ranker_result(items, {}, inverse=True)
        periods = period_stability(items)
        stability = stability_classification(periods)
        rolling = rolling_stability(items)
        results[name] = FeatureInformationResult(
            name=name,
            inventory=inventory[name],
            separation=separation,
            direct_ranker=direct,
            inverse_ranker=inverse,
            inverse_is_diagnostic=True,
            period_stability=periods,
            stability_classification=stability,
            rolling_stability=rolling,
            recommendation=_feature_recommendation(direct, stability),
        )
    return results


def ranker_result(
    snapshots: tuple[FeatureSnapshot, ...],
    endpoints: dict[str, RankEndpointResult],
    *,
    inverse: bool,
) -> RankerResult:
    ranks = tuple(
        rank
        for snapshot in snapshots
        for rank in (snapshot.inverse_winner_ranks if inverse else snapshot.winner_ranks)
    )
    random_ranks = tuple(rank for snapshot in snapshots for rank in snapshot.random_winner_ranks)
    return RankerResult(
        sample_size=len(snapshots),
        mean_winner_rank=mean(ranks) if ranks else 0.0,
        median_winner_rank=median(ranks) if ranks else 0.0,
        mean_best_winner_rank=mean(
            min(snapshot.inverse_winner_ranks if inverse else snapshot.winner_ranks)
            for snapshot in snapshots
        )
        if snapshots
        else 0.0,
        mean_worst_winner_rank=mean(
            max(snapshot.inverse_winner_ranks if inverse else snapshot.winner_ranks)
            for snapshot in snapshots
        )
        if snapshots
        else 0.0,
        top5_capture_rate=_capture_rate(snapshots, 5, inverse=inverse),
        top10_capture_rate=_capture_rate(snapshots, 10, inverse=inverse),
        top15_capture_rate=_capture_rate(snapshots, 15, inverse=inverse),
        top20_capture_rate=_capture_rate(snapshots, 20, inverse=inverse),
        random_mean_winner_rank=mean(random_ranks) if random_ranks else 0.0,
        random_top5_capture_rate=_random_capture_rate(snapshots, 5),
        random_top15_capture_rate=_random_capture_rate(snapshots, 15),
        primary_endpoints=endpoints,
    )


def period_stability(
    snapshots: tuple[FeatureSnapshot, ...],
) -> tuple[PeriodStabilityResult, ...]:
    periods = (
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024-1401", "2024-01-01", "9999-12-31"),
    )
    rows: list[PeriodStabilityResult] = []
    for label, start, end in periods:
        selected = tuple(item for item in snapshots if start <= item.draw_date <= end)
        if not selected:
            continue
        mean_rank = _mean_winner_rank(selected, inverse=False)
        random_rank = _mean_random_rank(selected)
        rows.append(
            PeriodStabilityResult(
                period=label,
                target_draws=len(selected),
                mean_winner_rank=mean_rank,
                random_mean_winner_rank=random_rank,
                mean_rank_advantage=random_rank - mean_rank,
                top5_capture_rate=_capture_rate(selected, 5, inverse=False),
                random_top5_capture_rate=_random_capture_rate(selected, 5),
                top15_capture_rate=_capture_rate(selected, 15, inverse=False),
                random_top15_capture_rate=_random_capture_rate(selected, 15),
            )
        )
    return tuple(rows)


def rolling_stability(
    snapshots: tuple[FeatureSnapshot, ...],
    window_size: int = STAGE26_ROLLING_WINDOW,
) -> RollingStabilityResult:
    if len(snapshots) < window_size:
        return RollingStabilityResult(window_size, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    rank_advantages: list[float] = []
    top15_advantages: list[float] = []
    for start in range(0, len(snapshots) - window_size + 1):
        window = snapshots[start : start + window_size]
        rank_advantages.append(_mean_random_rank(window) - _mean_winner_rank(window, inverse=False))
        top15_advantages.append(
            _capture_rate(window, 15, inverse=False) - _random_capture_rate(window, 15)
        )
    return RollingStabilityResult(
        window_size=window_size,
        window_count=len(rank_advantages),
        positive_mean_rank_advantage_fraction=sum(value > 0 for value in rank_advantages)
        / len(rank_advantages),
        positive_top15_advantage_fraction=sum(value > 0 for value in top15_advantages)
        / len(top15_advantages),
        median_mean_rank_advantage=median(rank_advantages),
        minimum_mean_rank_advantage=min(rank_advantages),
        maximum_mean_rank_advantage=max(rank_advantages),
        recent_mean_rank_advantage=rank_advantages[-1],
        recent_top15_advantage=top15_advantages[-1],
        longest_positive_mean_rank_streak=_longest_positive_streak(rank_advantages),
    )


def redundancy_analysis(
    snapshots: dict[str, tuple[FeatureSnapshot, ...]],
    feature_names: tuple[str, ...],
) -> RedundancyResult:
    correlations: list[tuple[str, str, float]] = []
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1 :]:
            rho = mean(
                _spearman(left_snap.ranks, right_snap.ranks)
                for left_snap, right_snap in zip(snapshots[left], snapshots[right], strict=True)
            )
            correlations.append((left, right, rho))
    high = tuple(
        sorted(
            (item for item in correlations if abs(item[2]) >= 0.9),
            key=lambda item: (-abs(item[2]), item[0], item[1]),
        )
    )
    independent = tuple(
        sorted(
            (item for item in correlations if abs(item[2]) <= 0.3),
            key=lambda item: (abs(item[2]), item[0], item[1]),
        )[:20]
    )
    pair_strength = tuple(
        sorted(
            (
                (right if left == "pair_strength_rate" else left, rho)
                for left, right, rho in correlations
                if "pair_strength_rate" in {left, right}
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )
    )
    return RedundancyResult(
        average_pairwise_spearman=mean(tuple(abs(item[2]) for item in correlations))
        if correlations
        else 0.0,
        high_redundancy_pairs=high,
        relatively_independent_pairs=independent,
        pair_strength_redundancy=pair_strength,
    )


def champion_attribution(
    discovery: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    feature_snapshots: tuple[FeatureSnapshot, ...],
    seed: int,
    min_training_draws: int,
) -> ChampionAttributionResult:
    blocks = build_walk_forward_feature_blocks(discovery, lottery, FEATURE_GROUPS["pair_only"])
    lr = evaluate_score_model(
        discovery,
        lottery,
        blocks,
        seed=seed,
        c_value=1.0,
        calibration="uncalibrated",
        min_training_draws=min_training_draws,
        refit_interval=DEFAULT_ML_REFIT_INTERVAL,
    )["snapshots"]
    equal = []
    spearman = []
    top5_equal = []
    top15_equal = []
    winner_rank_diffs = []
    top5_diffs = []
    top15_diffs = []
    for direct, model in zip(feature_snapshots, lr, strict=True):
        equal.append(direct.ranks == model.ranks)
        spearman.append(_spearman(direct.ranks, model.ranks))
        top5_equal.append(_top_set(direct.ranks, 5) == _top_set(model.ranks, 5))
        top15_equal.append(_top_set(direct.ranks, 15) == _top_set(model.ranks, 15))
        winner_rank_diffs.extend(
            direct.ranks[number] - model.ranks[number] for number in direct.winning_numbers
        )
        top5_diffs.append(
            sum(model.ranks[number] <= 5 for number in direct.winning_numbers) / 5
            - sum(direct.ranks[number] <= 5 for number in direct.winning_numbers) / 5
        )
        top15_diffs.append(
            sum(model.ranks[number] <= 15 for number in direct.winning_numbers) / 5
            - sum(direct.ranks[number] <= 15 for number in direct.winning_numbers) / 5
        )
    return ChampionAttributionResult(
        direct_feature="pair_strength_rate",
        lr_feature_group="pair_only",
        rank_order_equality_rate=mean(equal),
        average_spearman=mean(spearman),
        top5_membership_equality_rate=mean(top5_equal),
        top15_membership_equality_rate=mean(top15_equal),
        winner_rank_difference_mean=mean(winner_rank_diffs) if winner_rank_diffs else 0.0,
        top5_capture_difference=mean(top5_diffs) if top5_diffs else 0.0,
        top15_capture_difference=mean(top15_diffs) if top15_diffs else 0.0,
        interpretation=(
            "Logistic Regression adds little or no ranking transformation beyond "
            "pair_strength_rate ordering."
        ),
    )


def strongest_feature(results: dict[str, FeatureInformationResult]) -> dict[str, Any]:
    name, result = max(
        results.items(),
        key=lambda item: (
            item[1].direct_ranker.primary_endpoints["mean_winner_rank"].difference,
            item[1].direct_ranker.primary_endpoints["top15_capture_rate"].difference,
            -item[1].direct_ranker.primary_endpoints["mean_winner_rank"].holm_p_value,
            item[0],
        ),
    )
    primary = result.direct_ranker.primary_endpoints["mean_winner_rank"]
    return {
        "feature": name,
        "classification": primary.classification,
        "mean_winner_rank": result.direct_ranker.mean_winner_rank,
        "random_mean_winner_rank": result.direct_ranker.random_mean_winner_rank,
        "mean_rank_advantage": primary.difference,
        "top5_capture_rate": result.direct_ranker.top5_capture_rate,
        "top15_capture_rate": result.direct_ranker.top15_capture_rate,
        "raw_p_value": primary.raw_p_value,
        "holm_p_value": primary.holm_p_value,
        "bh_p_value": primary.bh_p_value,
        "stability_classification": result.stability_classification,
        "recommendation": result.recommendation,
    }


def stage27_feature_recommendation(
    results: dict[str, FeatureInformationResult],
    redundancy: RedundancyResult,
) -> str:
    candidates = tuple(
        name for name, result in results.items() if result.recommendation == "CANDIDATE_FOR_STAGE27"
    )
    if not candidates:
        return "NONE"
    redundant = {pair[1] for pair in redundancy.high_redundancy_pairs}
    selected = tuple(name for name in candidates if name not in redundant)
    return selected[0] if selected else "NONE"


def stage27_ensemble_recommendation(
    results: dict[str, FeatureInformationResult],
    redundancy: RedundancyResult,
) -> str:
    candidates = tuple(
        name for name, result in results.items() if result.recommendation == "CANDIDATE_FOR_STAGE27"
    )
    if len(candidates) < 2:
        return "NONE"
    high_pairs = {
        frozenset((left, right)) for left, right, _rho in redundancy.high_redundancy_pairs
    }
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if frozenset((left, right)) not in high_pairs:
                return f"{left}+{right}"
    return "NONE"


def frozen_decision_payload(
    discovery: tuple[HistoricalDraw, ...],
    inventory: dict[str, FeatureInventoryItem],
    strongest: dict[str, Any],
    feature_recommendation: str,
    ensemble_recommendation: str,
    redundancy: RedundancyResult,
    *,
    seed: int,
    min_training_draws: int,
    bootstrap_replications: int,
) -> dict[str, Any]:
    return {
        "schema_version": STAGE26_DECISION_SCHEMA_VERSION,
        "lottery": "MINI_LOTO",
        "discovery_cutoff_draw": STAGE26_DISCOVERY_CUTOFF_DRAW,
        "discovery_dataset_hash": calculate_dataset_hash(discovery),
        "discovery_draw_count": len(discovery),
        "feature_inventory": inventory,
        "confirmatory_features": FEATURE_NAMES_V2,
        "expected_directions": FEATURE_DIRECTIONS,
        "primary_endpoints": STAGE26_PRIMARY_ENDPOINTS,
        "multiplicity_correction": "Holm across feature x primary endpoint hypotheses",
        "bh_exploratory_reported": True,
        "strongest_feature": strongest,
        "redundancy_summary": {
            "average_pairwise_spearman": redundancy.average_pairwise_spearman,
            "high_redundancy_pairs": redundancy.high_redundancy_pairs,
            "pair_strength_redundancy": redundancy.pair_strength_redundancy,
        },
        "stage27_feature_recommendation": feature_recommendation,
        "stage27_ensemble_recommendation": ensemble_recommendation,
        "classification": strongest["classification"],
        "configuration": {
            "seed": seed,
            "min_training_draws": min_training_draws,
            "bootstrap_replications": bootstrap_replications,
            "rolling_window": STAGE26_ROLLING_WINDOW,
        },
        "frozen_before_holdout": True,
        "excluded_from_discovery": {
            "holdout_draw": STAGE26_HOLDOUT_DRAW,
            "later_draws": "all draws after #1402",
        },
    }


def evaluate_holdout_after_frozen_decision(
    all_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    frozen_decision: dict[str, Any],
    seed: int,
) -> HoldoutResult:
    if not frozen_decision.get("frozen_before_holdout"):
        raise ResearchValidationError("Stage 26 decision must be frozen before holdout")
    discovery = discovery_slice(all_draws, lottery)
    holdout = next((draw for draw in all_draws if draw.draw_number == STAGE26_HOLDOUT_DRAW), None)
    feature = str(frozen_decision["strongest_feature"]["feature"])
    blocks = build_walk_forward_feature_blocks(
        (*discovery, holdout) if holdout else discovery, lottery, FEATURE_NAMES_V2
    )
    feature_index = FEATURE_NAMES_V2.index(feature)
    block = blocks[-1]
    values = {row.number: float(row.features[feature_index]) for row in block.rows}
    ranks = rank_numbers_from_scores(_directional_scores(values, FEATURE_DIRECTIONS[feature]))
    top5 = tuple(number for number, rank in sorted(ranks.items(), key=lambda item: item[1])[:5])
    top15 = tuple(number for number, rank in sorted(ranks.items(), key=lambda item: item[1])[:15])
    if holdout is None:
        return HoldoutResult(False, None, None, feature, top5, top15, (), {}, 0, 0)
    return HoldoutResult(
        True,
        holdout.draw_number,
        holdout.draw_date.isoformat(),
        feature,
        top5,
        top15,
        holdout.main_numbers,
        {number: ranks[number] for number in holdout.main_numbers},
        sum(ranks[number] <= 5 for number in holdout.main_numbers),
        sum(ranks[number] <= 15 for number in holdout.main_numbers),
    )


def run_stage26_leakage_audit(
    discovery: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(min_training_draws, 2), len(discovery) - 2)
    blocks = build_walk_forward_feature_blocks(discovery, lottery, FEATURE_NAMES_V2)
    original = tuple(row.features for row in blocks[target_index].rows)
    target_mutation = list(discovery)
    target = target_mutation[target_index]
    target_mutation[target_index] = HistoricalDraw(
        target.lottery,
        target.draw_number,
        target.draw_date,
        tuple(reversed(target.main_numbers)),
        target.bonus_numbers,
    )
    target_features = tuple(
        row.features
        for row in build_walk_forward_feature_blocks(
            tuple(target_mutation), lottery, FEATURE_NAMES_V2
        )[target_index].rows
    )
    future_mutation = list(discovery)
    future = future_mutation[target_index + 1]
    future_mutation[target_index + 1] = HistoricalDraw(
        future.lottery,
        future.draw_number,
        future.draw_date,
        tuple(reversed(future.main_numbers)),
        future.bonus_numbers,
    )
    feature_name = "pair_strength_rate"
    original_snapshots = build_feature_snapshots(
        blocks[: target_index + 1],
        discovery[: target_index + 1],
        lottery,
        seed=seed,
        min_training_draws=target_index,
    )[feature_name]
    future_blocks = build_walk_forward_feature_blocks(
        tuple(future_mutation), lottery, FEATURE_NAMES_V2
    )
    future_snapshots = build_feature_snapshots(
        future_blocks[: target_index + 1],
        tuple(future_mutation[: target_index + 1]),
        lottery,
        seed=seed,
        min_training_draws=target_index,
    )[feature_name]
    training_dates = build_training_dataset(blocks, target_index)[2]
    training_ok = all(
        training_date < discovery[target_index].draw_date.isoformat()
        for training_date in training_dates
    )
    target_changed = original != target_features
    future_changed = original_snapshots[0].ranks != future_snapshots[0].ranks
    return LeakageAudit(
        lookahead_safe=training_ok and not target_changed and not future_changed,
        training_dates_strictly_before_target=training_ok,
        target_mutation_changes_features=target_changed,
        future_mutation_changes_prediction=future_changed,
    )


def stable_payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(research_result_json(payload).encode("utf-8")).hexdigest()


def save_stage26_outputs(
    result: Stage26FeatureInformationAudit,
    output_dir: str | Path = STAGE26_OUTPUT_DIR,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    decision_path = save_stage26_frozen_decision(result.frozen_decision, root)
    result_path = root / "v2_stage26_feature_information.json"
    summary_path = root / "v2_stage26_summary.json"
    result_path.write_text(research_result_json(result), encoding="utf-8")
    summary_path.write_text(
        research_result_json(
            {
                "schema_version": "v2-stage26-summary-v1",
                "lottery": result.lottery,
                "discovery_cutoff_draw": result.discovery_cutoff_draw,
                "discovery_dataset_hash": result.discovery_dataset_hash,
                "strongest_feature": result.strongest_feature,
                "stage27_feature_recommendation": result.stage27_feature_recommendation,
                "stage27_ensemble_recommendation": result.stage27_ensemble_recommendation,
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


def save_stage26_frozen_decision(
    frozen_decision: dict[str, Any],
    output_dir: str | Path = STAGE26_OUTPUT_DIR,
) -> Path:
    path = Path(output_dir) / "v2_stage26_frozen_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(frozen_decision), encoding="utf-8")
    return path


def _separation_result(
    snapshots: tuple[FeatureSnapshot, ...],
    direction: str,
    differences: tuple[float, ...],
    *,
    raw_p: float,
    holm_p: float,
    bh_p: float,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
) -> SeparationResult:
    winner_values = tuple(value for snapshot in snapshots for value in snapshot.winner_values)
    loser_values = tuple(value for snapshot in snapshots for value in snapshot.loser_values)
    diff = mean(differences) if differences else 0.0
    winner_mean = mean(winner_values) if winner_values else 0.0
    loser_mean = mean(loser_values) if loser_values else 0.0
    raw_difference = winner_mean - loser_mean
    if direction == "lower":
        diff = -raw_difference
    ci = _paired_mean_ci(
        differences,
        confidence_level=confidence_level,
    )
    effect = _effect_size(differences, loser_mean)
    return SeparationResult(
        winner_mean,
        loser_mean,
        raw_difference,
        diff,
        effect,
        ci,
        raw_p,
        holm_p,
        bh_p,
        _classification(diff, ci, holm_p),
    )


def _rank_endpoint_result(
    snapshots: tuple[FeatureSnapshot, ...],
    endpoint: str,
    differences: tuple[float, ...],
    *,
    raw_p: float,
    holm_p: float,
    bh_p: float,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
) -> RankEndpointResult:
    feature_values = tuple(
        _endpoint_value(snapshot, endpoint, random_baseline=False) for snapshot in snapshots
    )
    random_values = tuple(
        _endpoint_value(snapshot, endpoint, random_baseline=True) for snapshot in snapshots
    )
    ci = _paired_mean_ci(
        differences,
        confidence_level=confidence_level,
    )
    diff = mean(differences) if differences else 0.0
    return RankEndpointResult(
        endpoint,
        mean(feature_values) if feature_values else 0.0,
        mean(random_values) if random_values else 0.0,
        diff,
        ci,
        raw_p,
        holm_p,
        bh_p,
        _effect_size(differences, mean(random_values) if random_values else 0.0),
        _classification(diff, ci, holm_p),
    )


def _separation_differences(
    snapshots: tuple[FeatureSnapshot, ...],
    direction: str,
) -> tuple[float, ...]:
    multiplier = 1 if direction == "higher" else -1
    return tuple(
        multiplier * (mean(snapshot.winner_values) - mean(snapshot.loser_values))
        for snapshot in snapshots
    )


def _endpoint_differences(
    snapshots: tuple[FeatureSnapshot, ...],
    endpoint: str,
) -> tuple[float, ...]:
    if endpoint == "mean_winner_rank":
        return tuple(
            _endpoint_value(snapshot, endpoint, random_baseline=True)
            - _endpoint_value(snapshot, endpoint, random_baseline=False)
            for snapshot in snapshots
        )
    return tuple(
        _endpoint_value(snapshot, endpoint, random_baseline=False)
        - _endpoint_value(snapshot, endpoint, random_baseline=True)
        for snapshot in snapshots
    )


def _endpoint_value(
    snapshot: FeatureSnapshot,
    endpoint: str,
    *,
    random_baseline: bool,
) -> float:
    ranks = snapshot.random_winner_ranks if random_baseline else snapshot.winner_ranks
    if endpoint == "mean_winner_rank":
        return mean(ranks)
    if endpoint == "top5_capture_rate":
        return sum(rank <= 5 for rank in ranks) / len(ranks)
    if endpoint == "top15_capture_rate":
        return sum(rank <= 15 for rank in ranks) / len(ranks)
    raise ResearchValidationError(f"unknown Stage 26 endpoint: {endpoint}")


def _directional_scores(values: dict[int, float], direction: str) -> dict[int, float]:
    if direction == "higher":
        return values
    if direction == "lower":
        return {number: -value for number, value in values.items()}
    raise ResearchValidationError(f"unknown feature direction: {direction}")


def _random_ranks(
    lottery: LotteryDefinition,
    seed: int,
    feature_name: str,
    draw_number: int,
) -> dict[int, int]:
    rng = random.Random(_derived_seed(seed, f"stage26-{feature_name}-{draw_number}"))
    order = rng.sample(
        range(lottery.number_min, lottery.number_max + 1),
        lottery.number_max - lottery.number_min + 1,
    )
    return {number: index + 1 for index, number in enumerate(order)}


def _capture_rate(
    snapshots: tuple[FeatureSnapshot, ...],
    cutoff: int,
    *,
    inverse: bool,
) -> float:
    ranks = tuple(
        rank
        for snapshot in snapshots
        for rank in (snapshot.inverse_winner_ranks if inverse else snapshot.winner_ranks)
    )
    return sum(rank <= cutoff for rank in ranks) / len(ranks) if ranks else 0.0


def _random_capture_rate(snapshots: tuple[FeatureSnapshot, ...], cutoff: int) -> float:
    ranks = tuple(rank for snapshot in snapshots for rank in snapshot.random_winner_ranks)
    return sum(rank <= cutoff for rank in ranks) / len(ranks) if ranks else 0.0


def _mean_winner_rank(snapshots: tuple[FeatureSnapshot, ...], *, inverse: bool) -> float:
    ranks = tuple(
        rank
        for snapshot in snapshots
        for rank in (snapshot.inverse_winner_ranks if inverse else snapshot.winner_ranks)
    )
    return mean(ranks) if ranks else 0.0


def _mean_random_rank(snapshots: tuple[FeatureSnapshot, ...]) -> float:
    ranks = tuple(rank for snapshot in snapshots for rank in snapshot.random_winner_ranks)
    return mean(ranks) if ranks else 0.0


def _top_set(ranks: dict[int, int], cutoff: int) -> set[int]:
    return {number for number, rank in ranks.items() if rank <= cutoff}


def _spearman(left: dict[int, int], right: dict[int, int]) -> float:
    numbers = tuple(sorted(left))
    n = len(numbers)
    if n < 2:
        return 0.0
    squared = sum((left[number] - right[number]) ** 2 for number in numbers)
    return 1 - (6 * squared) / (n * (n * n - 1))


def _classification(difference: float, ci: ConfidenceInterval, adjusted_p: float) -> str:
    if ci.upper < 0 and adjusted_p < 0.05:
        return "NEGATIVE"
    if difference < 0 and ci.upper <= 0:
        return "NEGATIVE"
    if ci.lower > 0 and adjusted_p < 0.05:
        return "EVIDENCE"
    if difference > 0 and adjusted_p < 0.1:
        return "WEAK_SIGNAL"
    if adjusted_p >= 0.1 and ci.lower <= 0 <= ci.upper:
        return "NO_EVIDENCE"
    return "INCONCLUSIVE"


def stability_classification(periods: tuple[PeriodStabilityResult, ...]) -> str:
    advantages = tuple(period.mean_rank_advantage for period in periods)
    if not advantages or all(abs(value) < 1e-12 for value in advantages):
        return "NO_EFFECT"
    positive = sum(value > 0 for value in advantages)
    negative = sum(value < 0 for value in advantages)
    if positive and negative:
        return "SIGN_REVERSAL"
    if positive == len(advantages):
        return "CONSISTENT"
    if positive == 1 and advantages[-1] > 0:
        return "RECENT_ONLY"
    if positive == 1 and advantages[0] > 0:
        return "EARLY_ONLY"
    return "MIXED"


def _feature_recommendation(ranker: RankerResult, stability: str) -> str:
    primary = ranker.primary_endpoints["mean_winner_rank"]
    top5 = ranker.primary_endpoints["top5_capture_rate"]
    top15 = ranker.primary_endpoints["top15_capture_rate"]
    if (
        primary.classification in {"EVIDENCE", "WEAK_SIGNAL"}
        and primary.difference > 0
        and top5.difference >= 0
        and top15.difference >= 0
        and stability == "CONSISTENT"
        and abs(primary.difference) >= 0.25
    ):
        return "CANDIDATE_FOR_STAGE27"
    return "NONE"


def _longest_positive_streak(values: list[float]) -> int:
    best = 0
    current = 0
    for value in values:
        if value > 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    std = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=0.0 if std == 0 else absolute / std,
    )


def _paired_mean_ci(
    differences: tuple[float, ...],
    *,
    confidence_level: float,
) -> ConfidenceInterval:
    diff = mean(differences) if differences else 0.0
    if len(differences) <= 1:
        return ConfidenceInterval(confidence_level, diff, diff)
    std = pstdev(differences)
    if std == 0:
        return ConfidenceInterval(confidence_level, diff, diff)
    # Normal approximation is used here to keep this broad feature-inventory audit lightweight.
    z_value = 1.959963984540054 if confidence_level == DEFAULT_CONFIDENCE_LEVEL else 1.96
    margin = z_value * std / (len(differences) ** 0.5)
    return ConfidenceInterval(confidence_level, diff - margin, diff + margin)


def _paired_mean_p_value(differences: tuple[float, ...]) -> float:
    if len(differences) <= 1:
        return 1.0
    diff = mean(differences)
    std = pstdev(differences)
    if std == 0:
        return 1.0 if diff == 0 else 0.0
    z_score = abs(diff) / (std / (len(differences) ** 0.5))
    return max(0.0, min(1.0, math.erfc(z_score / (2**0.5))))


def _required_lookback(name: str) -> str:
    if name.endswith("_5") or "_5_" in name:
        return "previous 5 draws plus accumulated history where needed"
    if name.endswith("_10") or "_10_" in name or name == "recent_activity_10":
        return "previous 10 draws plus accumulated history where needed"
    if name.endswith("_20") or "_20" in name:
        return "previous 20 draws plus accumulated history where needed"
    if name.endswith("_50") or "_50" in name:
        return "previous 50 draws plus accumulated history where needed"
    if name.endswith("_100"):
        return "previous 100 draws plus accumulated history where needed"
    if name.startswith("previous_draw"):
        return "immediately previous draw plus accumulated history"
    return "all prior draws available before target"
