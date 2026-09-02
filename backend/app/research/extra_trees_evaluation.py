from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import sklearn
from sklearn.ensemble import ExtraTreesClassifier

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
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTH,
    FEATURE_VERSION_V2,
    LeakageAudit,
    _derived_seed,
    _make_model,
    _scores_from_fitted_model,
    build_training_dataset,
    build_walk_forward_feature_blocks,
    run_leakage_audit,
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

STAGE20_SCHEMA_VERSION = "v2-stage20-extra-trees-v1"
EXPERIMENT_LEDGER_SCHEMA_VERSION = "v2-experiment-ledger-v1"
DEFAULT_EXPERIMENT_LEDGER_PATH = Path("data/exports/experiments/v2_experiment_ledger.json")
EXTRA_TREES_MODEL_NAME = "extra_trees"
CHALLENGER_KEEP = "KEEP_AS_CHALLENGER"
CHALLENGER_RETIRE = "RETIRE"
CHALLENGER_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class Stage20MetricComparison:
    model_value: float
    baseline_value: float
    difference: float
    difference_ci: ConfidenceInterval
    effect_size: EffectSize
    raw_p_value: float
    stage_adjusted_p_value: float
    ledger_adjusted_p_value: float
    bh_exploratory_p_value: float


@dataclass(frozen=True, slots=True)
class Stage20ModelResult:
    model_name: str
    model_parameters: dict[str, Any]
    feature_group: str
    feature_names: tuple[str, ...]
    sample_size: int
    tickets_per_draw: int
    mean_matches: float
    random_mean_matches: float
    comparison_vs_random: Stage20MetricComparison
    comparison_vs_champion: Stage20MetricComparison | None
    prize_qualified_rate: float
    period_stability: tuple[PeriodStability, ...]
    conclusion: str
    experiment_id: str
    experiment_status: str


@dataclass(frozen=True, slots=True)
class Stage20ExtraTreesResult:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    feature_version: str
    feature_group: str
    portfolio_method: str
    configuration: dict[str, Any]
    sklearn_version: str
    current_champion: Stage20ModelResult
    extra_trees: Stage20ModelResult
    leakage: LeakageAudit
    experiment_ledger: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ModelObservation:
    target_draw_number: int
    target_draw_date: str
    model_mean_matches: float
    random_mean_matches: float
    model_prize_rate: float
    random_prize_rate: float
    match_counts: tuple[int, ...]
    random_match_counts: tuple[int, ...]


def run_stage20_extra_trees_evaluation(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = DEFAULT_TICKETS_PER_DRAW,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    ml_refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
    experiment_ledger_path: str | Path = DEFAULT_EXPERIMENT_LEDGER_PATH,
) -> Stage20ExtraTreesResult:
    if tickets_per_draw <= 0:
        raise ResearchValidationError("tickets_per_draw must be positive")
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")
    if ml_min_training_draws < 2:
        raise ResearchValidationError("ml_min_training_draws must be at least 2")
    if ml_refit_interval <= 0:
        raise ResearchValidationError("ml_refit_interval must be positive")

    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    preflight = preflight_validate_benchmark_dataset(draws, lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough draws for Stage 20 Extra Trees evaluation")

    champion_name, feature_group, feature_names = _selected_stage20_config(lottery)
    blocks = build_walk_forward_feature_blocks(ordered, lottery, feature_names)
    observations = {
        champion_name: _walk_forward_observations(
            blocks,
            ordered,
            lottery,
            champion_name,
            seed=seed,
            tickets_per_draw=tickets_per_draw,
            ml_min_training_draws=ml_min_training_draws,
            ml_refit_interval=ml_refit_interval,
        ),
        EXTRA_TREES_MODEL_NAME: _walk_forward_observations(
            blocks,
            ordered,
            lottery,
            EXTRA_TREES_MODEL_NAME,
            seed=seed,
            tickets_per_draw=tickets_per_draw,
            ml_min_training_draws=ml_min_training_draws,
            ml_refit_interval=ml_refit_interval,
        ),
    }
    random_values = tuple(item.random_mean_matches for item in observations[champion_name])
    raw_stage_p_values = {
        f"{model_name}_vs_random": paired_permutation_p_value(
            tuple(
                item.model_mean_matches - item.random_mean_matches for item in model_observations
            ),
            seed=_derived_seed(seed, f"stage20-{lottery.code}-{model_name}-random"),
            replications=bootstrap_replications,
        )
        for model_name, model_observations in observations.items()
    }
    raw_stage_p_values[f"{EXTRA_TREES_MODEL_NAME}_vs_champion"] = paired_permutation_p_value(
        tuple(
            extra.model_mean_matches - champion.model_mean_matches
            for extra, champion in zip(
                observations[EXTRA_TREES_MODEL_NAME],
                observations[champion_name],
                strict=True,
            )
        ),
        seed=_derived_seed(seed, f"stage20-{lottery.code}-extra-champion"),
        replications=bootstrap_replications,
    )
    stage_adjusted = holm_adjust_p_values(raw_stage_p_values)
    raw_bh = benjamini_hochberg_adjust_p_values(raw_stage_p_values)

    leakage = run_leakage_audit(
        ordered,
        lottery,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 20 leakage audit failed")

    champion_result = _model_result(
        champion_name,
        observations[champion_name],
        lottery,
        feature_group=feature_group,
        feature_names=feature_names,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
        raw_p_value=raw_stage_p_values[f"{champion_name}_vs_random"],
        stage_adjusted_p_value=stage_adjusted[f"{champion_name}_vs_random"],
        bh_exploratory_p_value=raw_bh[f"{champion_name}_vs_random"],
        ledger_adjusted_p_value=stage_adjusted[f"{champion_name}_vs_random"],
        random_values=random_values,
        comparison_vs_champion=None,
        experiment_status="CURRENT_CHAMPION",
        dataset_hash=preflight.dataset_hash,
    )
    placeholder_extra = _model_result(
        EXTRA_TREES_MODEL_NAME,
        observations[EXTRA_TREES_MODEL_NAME],
        lottery,
        feature_group=feature_group,
        feature_names=feature_names,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
        raw_p_value=raw_stage_p_values[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        stage_adjusted_p_value=stage_adjusted[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        bh_exploratory_p_value=raw_bh[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        ledger_adjusted_p_value=stage_adjusted[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        random_values=random_values,
        comparison_vs_champion=_comparison_vs_champion(
            observations[EXTRA_TREES_MODEL_NAME],
            observations[champion_name],
            seed=seed,
            bootstrap_replications=bootstrap_replications,
            confidence_level=confidence_level,
            raw_p_value=raw_stage_p_values[f"{EXTRA_TREES_MODEL_NAME}_vs_champion"],
            stage_adjusted_p_value=stage_adjusted[f"{EXTRA_TREES_MODEL_NAME}_vs_champion"],
            bh_exploratory_p_value=raw_bh[f"{EXTRA_TREES_MODEL_NAME}_vs_champion"],
        ),
        experiment_status=CHALLENGER_INCONCLUSIVE,
        dataset_hash=preflight.dataset_hash,
    )
    ledger = register_stage20_experiment(
        placeholder_extra,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        ledger_path=experiment_ledger_path,
    )
    ledger_adjusted = _ledger_adjusted_for_experiment(
        ledger,
        placeholder_extra.experiment_id,
        fallback=placeholder_extra.comparison_vs_random.stage_adjusted_p_value,
    )
    ledger_bh = _ledger_bh_for_experiment(
        ledger,
        placeholder_extra.experiment_id,
        fallback=placeholder_extra.comparison_vs_random.bh_exploratory_p_value,
    )
    extra_result = _model_result(
        EXTRA_TREES_MODEL_NAME,
        observations[EXTRA_TREES_MODEL_NAME],
        lottery,
        feature_group=feature_group,
        feature_names=feature_names,
        seed=seed,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
        raw_p_value=raw_stage_p_values[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        stage_adjusted_p_value=stage_adjusted[f"{EXTRA_TREES_MODEL_NAME}_vs_random"],
        bh_exploratory_p_value=ledger_bh,
        ledger_adjusted_p_value=ledger_adjusted,
        random_values=random_values,
        comparison_vs_champion=placeholder_extra.comparison_vs_champion,
        experiment_status=_challenger_status(placeholder_extra, champion_result, ledger_adjusted),
        dataset_hash=preflight.dataset_hash,
    )
    ledger = register_stage20_experiment(
        extra_result,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        ledger_path=experiment_ledger_path,
    )
    return Stage20ExtraTreesResult(
        schema_version=STAGE20_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        dataset_range={
            "first_draw_number": preflight.first_draw_number,
            "last_draw_number": preflight.last_draw_number,
            "first_draw_date": preflight.first_draw_date,
            "last_draw_date": preflight.last_draw_date,
            "draw_count": preflight.draw_count,
        },
        feature_version=FEATURE_VERSION_V2,
        feature_group=feature_group,
        portfolio_method="top_ranked",
        configuration={
            "seed": seed,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "tickets_per_draw": tickets_per_draw,
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": ml_refit_interval,
            "decision_rule": (
                "keep Extra Trees only when it is positive vs random and champion, "
                "ledger-adjusted evidence is stronger than the champion, "
                "and stability is acceptable"
            ),
        },
        sklearn_version=sklearn.__version__,
        current_champion=champion_result,
        extra_trees=extra_result,
        leakage=leakage,
        experiment_ledger={
            "path": str(Path(experiment_ledger_path)),
            "schema_version": ledger.get("schema_version"),
            "hypothesis_count": len(ledger.get("entries", ())),
            "correction_scope": "all entries in the V2 experiment ledger",
            "multiple_comparison_method": "Holm; BH exploratory reported separately",
        },
        warnings=(
            "Stage 20 is historical research only and does not alter production strategy.",
            "Extra Trees is not promoted by this evaluation.",
            "ML scores are ranking values, not winning probabilities.",
        ),
    )


def make_extra_trees_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=DEFAULT_RF_ESTIMATORS,
        max_depth=DEFAULT_RF_MAX_DEPTH,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def extra_trees_model_parameters(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": DEFAULT_RF_ESTIMATORS,
        "max_depth": DEFAULT_RF_MAX_DEPTH,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": 1,
    }


def benjamini_hochberg_adjust_p_values(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1], reverse=True)
    adjusted: dict[str, float] = {}
    running_min = 1.0
    total = len(ordered)
    for reverse_rank, (name, p_value) in enumerate(ordered, start=1):
        rank = total - reverse_rank + 1
        adjusted_value = min(running_min, min(1.0, p_value * total / rank))
        running_min = adjusted_value
        adjusted[name] = adjusted_value
    return adjusted


def register_stage20_experiment(
    result: Stage20ModelResult,
    *,
    lottery: str,
    dataset_hash: str,
    ledger_path: str | Path = DEFAULT_EXPERIMENT_LEDGER_PATH,
) -> dict[str, Any]:
    path = Path(ledger_path)
    ledger = _load_experiment_ledger(path)
    entry = {
        "experiment_id": result.experiment_id,
        "stage": "20",
        "lottery": lottery,
        "dataset_hash": dataset_hash,
        "hypothesis": f"Extra Trees may improve historical {result.feature_group} ranking",
        "model": result.model_name,
        "feature_group": result.feature_group,
        "comparison": "extra_trees_vs_random",
        "raw_p_value": result.comparison_vs_random.raw_p_value,
        "stage_adjusted_p_value": result.comparison_vs_random.stage_adjusted_p_value,
        "ledger_adjusted_p_value": result.comparison_vs_random.ledger_adjusted_p_value,
        "bh_exploratory_p_value": result.comparison_vs_random.bh_exploratory_p_value,
        "conclusion": result.conclusion,
        "status": result.experiment_status,
    }
    existing = {
        str(item["experiment_id"]): item
        for item in ledger.get("entries", ())
        if "experiment_id" in item
    }
    existing[result.experiment_id] = entry
    entries = sorted(existing.values(), key=lambda item: str(item["experiment_id"]))
    holm = holm_adjust_p_values(
        {str(item["experiment_id"]): float(item["raw_p_value"]) for item in entries}
    )
    bh = benjamini_hochberg_adjust_p_values(
        {str(item["experiment_id"]): float(item["raw_p_value"]) for item in entries}
    )
    for item in entries:
        experiment_id = str(item["experiment_id"])
        item["ledger_adjusted_p_value"] = holm[experiment_id]
        item["bh_exploratory_p_value"] = bh[experiment_id]
    payload = {
        "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
        "multiple_comparison_method": "holm",
        "exploratory_method": "benjamini_hochberg",
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return payload


def save_stage20_extra_trees_result(
    result: Stage20ExtraTreesResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def sync_stage20_result_file_with_ledger(
    result_path: str | Path,
    ledger_path: str | Path = DEFAULT_EXPERIMENT_LEDGER_PATH,
) -> dict[str, Any]:
    path = Path(result_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = payload["extra_trees"]["experiment_id"]
    ledger = _load_experiment_ledger(Path(ledger_path))
    for entry in ledger.get("entries", ()):
        if entry.get("experiment_id") != experiment_id:
            continue
        payload["extra_trees"]["comparison_vs_random"]["ledger_adjusted_p_value"] = entry[
            "ledger_adjusted_p_value"
        ]
        payload["extra_trees"]["comparison_vs_random"]["bh_exploratory_p_value"] = entry[
            "bh_exploratory_p_value"
        ]
        payload["extra_trees"]["experiment_status"] = entry["status"]
        path.write_text(research_result_json(payload), encoding="utf-8")
        return payload
    return payload


def _selected_stage20_config(lottery: LotteryDefinition) -> tuple[str, str, tuple[str, ...]]:
    if str(lottery.code) == "LOTO6":
        return "random_forest", "gap_only", FEATURE_GROUPS["gap_only"]
    if str(lottery.code) == "MINI_LOTO":
        return "logistic_regression", "pair_only", FEATURE_GROUPS["pair_only"]
    raise ResearchValidationError(f"unsupported lottery for Stage 20: {lottery.code}")


def _walk_forward_observations(
    blocks,
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    model_name: str,
    *,
    seed: int,
    tickets_per_draw: int,
    ml_min_training_draws: int,
    ml_refit_interval: int,
) -> tuple[_ModelObservation, ...]:
    observations: list[_ModelObservation] = []
    model = None
    last_fit_index: int | None = None
    for target_index in range(ml_min_training_draws, len(blocks)):
        target_draw = draws[target_index]
        if (
            model is None
            or last_fit_index is None
            or target_index - last_fit_index >= ml_refit_interval
        ):
            x_train, y_train, training_dates = build_training_dataset(blocks, target_index)
            if not all(
                date_text < target_draw.draw_date.isoformat() for date_text in training_dates
            ):
                raise ResearchValidationError("training data includes target/future draw date")
            model = _make_stage20_model(model_name, seed)
            model.fit(x_train, y_train)
            last_fit_index = target_index
        scores = _scores_from_fitted_model(model, blocks[target_index])
        model_tickets = _top_ranked_tickets(scores, lottery, tickets_per_draw)
        random_tickets = generate_distinct_random_tickets(
            lottery,
            random.Random(seed + target_draw.draw_number),
            tickets_per_draw,
        )
        observations.append(
            _evaluate_observation(
                target_draw,
                lottery,
                model_tickets=model_tickets,
                random_tickets=random_tickets,
            )
        )
    return tuple(observations)


def _make_stage20_model(model_name: str, seed: int):
    if model_name == EXTRA_TREES_MODEL_NAME:
        return make_extra_trees_model(seed)
    return _make_model(model_name, seed)


def _top_ranked_tickets(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    tickets_per_draw: int,
) -> tuple[tuple[int, ...], ...]:
    ranked = tuple(
        number for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )
    ticket_size = lottery.numbers_per_ticket
    needed = ticket_size * tickets_per_draw
    if len(ranked) < needed:
        raise ResearchValidationError("not enough scored numbers to construct Stage 20 portfolio")
    tickets = tuple(
        lottery.validate_main_numbers(ranked[index : index + ticket_size])
        for index in range(0, needed, ticket_size)
    )
    if len(set(tickets)) != len(tickets):
        raise ResearchValidationError("Stage 20 top-ranked portfolio produced duplicate tickets")
    return tickets


def _evaluate_observation(
    target_draw: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    model_tickets: tuple[tuple[int, ...], ...],
    random_tickets: tuple[tuple[int, ...], ...],
) -> _ModelObservation:
    model_results = tuple(match_ticket(ticket, target_draw, lottery) for ticket in model_tickets)
    random_results = tuple(match_ticket(ticket, target_draw, lottery) for ticket in random_tickets)
    model_matches = tuple(result.main_match_count for result in model_results)
    random_matches = tuple(result.main_match_count for result in random_results)
    return _ModelObservation(
        target_draw_number=target_draw.draw_number,
        target_draw_date=target_draw.draw_date.isoformat(),
        model_mean_matches=mean(model_matches),
        random_mean_matches=mean(random_matches),
        model_prize_rate=sum(result.qualifies_for_prize for result in model_results)
        / len(model_results),
        random_prize_rate=sum(result.qualifies_for_prize for result in random_results)
        / len(random_results),
        match_counts=model_matches,
        random_match_counts=random_matches,
    )


def _model_result(
    model_name: str,
    observations: tuple[_ModelObservation, ...],
    lottery: LotteryDefinition,
    *,
    feature_group: str,
    feature_names: tuple[str, ...],
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
    raw_p_value: float,
    stage_adjusted_p_value: float,
    ledger_adjusted_p_value: float,
    bh_exploratory_p_value: float,
    random_values: tuple[float, ...],
    comparison_vs_champion: Stage20MetricComparison | None,
    experiment_status: str,
    dataset_hash: str,
) -> Stage20ModelResult:
    model_values = tuple(observation.model_mean_matches for observation in observations)
    differences = tuple(
        left - right for left, right in zip(model_values, random_values, strict=True)
    )
    diff_ci = bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, f"stage20-{lottery.code}-{model_name}-random-ci"),
        replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    effect = _effect_size(differences, mean(random_values) if random_values else 0.0)
    periods = _period_stability(observations)
    positive_periods = sum(period.mean_match_difference > 0 for period in periods)
    conclusion = classify_conclusion(
        adjusted_p_value=ledger_adjusted_p_value,
        difference_ci=diff_ci,
        standardized_effect=effect.standardized_mean_difference,
        stable_positive_periods=positive_periods,
        total_periods=len(periods),
    )
    return Stage20ModelResult(
        model_name=model_name,
        model_parameters=(
            extra_trees_model_parameters(seed)
            if model_name == EXTRA_TREES_MODEL_NAME
            else _champion_parameters(model_name, seed)
        ),
        feature_group=feature_group,
        feature_names=feature_names,
        sample_size=len(observations),
        tickets_per_draw=len(observations[0].match_counts) if observations else 0,
        mean_matches=mean(model_values) if model_values else 0.0,
        random_mean_matches=mean(random_values) if random_values else 0.0,
        comparison_vs_random=Stage20MetricComparison(
            model_value=mean(model_values) if model_values else 0.0,
            baseline_value=mean(random_values) if random_values else 0.0,
            difference=mean(differences) if differences else 0.0,
            difference_ci=diff_ci,
            effect_size=effect,
            raw_p_value=raw_p_value,
            stage_adjusted_p_value=stage_adjusted_p_value,
            ledger_adjusted_p_value=ledger_adjusted_p_value,
            bh_exploratory_p_value=bh_exploratory_p_value,
        ),
        comparison_vs_champion=comparison_vs_champion,
        prize_qualified_rate=mean(tuple(item.model_prize_rate for item in observations))
        if observations
        else 0.0,
        period_stability=periods,
        conclusion=conclusion,
        experiment_id=deterministic_stage20_experiment_id(
            lottery=str(lottery.code),
            dataset_hash=dataset_hash,
            model_name=model_name,
            feature_group=feature_group,
            seed=seed,
            bootstrap_replications=bootstrap_replications,
            tickets_per_draw=len(observations[0].match_counts) if observations else 0,
        ),
        experiment_status=experiment_status,
    )


def _comparison_vs_champion(
    extra_observations: tuple[_ModelObservation, ...],
    champion_observations: tuple[_ModelObservation, ...],
    *,
    seed: int,
    bootstrap_replications: int,
    confidence_level: float,
    raw_p_value: float,
    stage_adjusted_p_value: float,
    bh_exploratory_p_value: float,
) -> Stage20MetricComparison:
    extra_values = tuple(observation.model_mean_matches for observation in extra_observations)
    champion_values = tuple(observation.model_mean_matches for observation in champion_observations)
    differences = tuple(
        left - right for left, right in zip(extra_values, champion_values, strict=True)
    )
    diff_ci = bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, "stage20-extra-vs-champion-ci"),
        replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    return Stage20MetricComparison(
        model_value=mean(extra_values) if extra_values else 0.0,
        baseline_value=mean(champion_values) if champion_values else 0.0,
        difference=mean(differences) if differences else 0.0,
        difference_ci=diff_ci,
        effect_size=_effect_size(differences, mean(champion_values) if champion_values else 0.0),
        raw_p_value=raw_p_value,
        stage_adjusted_p_value=stage_adjusted_p_value,
        ledger_adjusted_p_value=stage_adjusted_p_value,
        bh_exploratory_p_value=bh_exploratory_p_value,
    )


def deterministic_stage20_experiment_id(
    *,
    lottery: str,
    dataset_hash: str,
    model_name: str,
    feature_group: str,
    seed: int,
    bootstrap_replications: int,
    tickets_per_draw: int,
) -> str:
    payload = "|".join(
        (
            STAGE20_SCHEMA_VERSION,
            lottery,
            dataset_hash,
            model_name,
            feature_group,
            str(seed),
            str(bootstrap_replications),
            str(tickets_per_draw),
        )
    )
    return "V2-EXP-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _challenger_status(
    extra_result: Stage20ModelResult,
    champion_result: Stage20ModelResult,
    ledger_adjusted_p_value: float,
) -> str:
    comparison = extra_result.comparison_vs_champion
    if comparison is None:
        return CHALLENGER_INCONCLUSIVE
    stable_positive = sum(
        period.mean_match_difference > 0 for period in extra_result.period_stability
    )
    has_stability_issue = stable_positive < max(1, len(extra_result.period_stability) - 1)
    if (
        extra_result.comparison_vs_random.difference > 0
        and comparison.difference > 0
        and ledger_adjusted_p_value < champion_result.comparison_vs_random.ledger_adjusted_p_value
        and not has_stability_issue
    ):
        return CHALLENGER_KEEP
    return CHALLENGER_RETIRE


def _champion_parameters(model_name: str, seed: int) -> dict[str, Any]:
    if model_name == "random_forest":
        return {
            "n_estimators": DEFAULT_RF_ESTIMATORS,
            "max_depth": DEFAULT_RF_MAX_DEPTH,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "random_state": seed,
            "n_jobs": 1,
        }
    return {
        "class_weight": "balanced",
        "max_iter": 250,
        "random_state": seed,
        "solver": "liblinear",
    }


def _period_stability(observations: tuple[_ModelObservation, ...]) -> tuple[PeriodStability, ...]:
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
            observation.model_mean_matches - observation.random_mean_matches
            for observation in period_observations
        )
        prize_diff = mean(
            observation.model_prize_rate - observation.random_prize_rate
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


def _load_experiment_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": EXPERIMENT_LEDGER_SCHEMA_VERSION,
            "multiple_comparison_method": "holm",
            "exploratory_method": "benjamini_hochberg",
            "entries": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_adjusted_for_experiment(
    ledger: dict[str, Any],
    experiment_id: str,
    *,
    fallback: float,
) -> float:
    for entry in ledger.get("entries", ()):
        if entry.get("experiment_id") == experiment_id:
            return float(entry.get("ledger_adjusted_p_value", fallback))
    return fallback


def _ledger_bh_for_experiment(
    ledger: dict[str, Any],
    experiment_id: str,
    *,
    fallback: float,
) -> float:
    for entry in ledger.get("entries", ()):
        if entry.get("experiment_id") == experiment_id:
            return float(entry.get("bh_exploratory_p_value", fallback))
    return fallback
