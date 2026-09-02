from __future__ import annotations

import hashlib
import json
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
    generate_distinct_random_tickets,
    preflight_validate_benchmark_dataset,
)
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import (
    DEFAULT_EXPERIMENT_LEDGER_PATH,
    benjamini_hochberg_adjust_p_values,
)
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    DrawFeatureBlock,
    LeakageAudit,
    NumberFeatureRow,
    _derived_seed,
    _make_model,
    _scores_from_fitted_model,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json
from backend.app.research.prize import match_ticket
from backend.app.research.production import _generate_ranked_tickets
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

STAGE22_SCHEMA_VERSION = "v2-stage22-pair-network-v1"
PAIR_NETWORK_FEATURE_GROUP = "pair_network_v1"
PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW = 1400
PAIR_NETWORK_RECENT_WINDOW = 20
PAIR_NETWORK_NEW_FEATURES = (
    "weighted_degree",
    "normalized_weighted_degree",
    "neighbor_strength_mean",
    "neighbor_strength_max",
    "recent_pair_strength",
)
PAIR_NETWORK_FEATURE_NAMES = (*FEATURE_GROUPS["pair_only"], *PAIR_NETWORK_NEW_FEATURES)
VERDICT_KEEP = "KEEP_AS_SHADOW_CANDIDATE"
VERDICT_RETIRE = "RETIRE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class PairNetwork:
    draw_count: int
    pair_counts: Counter[tuple[int, int]]
    recent_pair_counts: Counter[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class Stage22Comparison:
    challenger_value: float
    baseline_value: float
    difference: float
    difference_ci: ConfidenceInterval
    effect_size: EffectSize
    raw_p_value: float


@dataclass(frozen=True, slots=True)
class WinTieLoss:
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True, slots=True)
class Stage22Result:
    schema_version: str
    lottery: str
    dataset_hash: str
    dataset_range: dict[str, str | int]
    preregistration: dict[str, Any]
    pair_only_baseline_features: tuple[str, ...]
    pair_network_features: tuple[str, ...]
    new_pair_network_features: tuple[str, ...]
    model: str
    model_parameters: dict[str, Any]
    portfolio_method: str
    configuration: dict[str, Any]
    sklearn_version: str
    champion: dict[str, Any]
    challenger: dict[str, Any]
    challenger_vs_champion: Stage22Comparison
    challenger_vs_random: Stage22Comparison
    win_tie_loss: WinTieLoss
    period_stability: tuple[PeriodStability, ...]
    leakage: LeakageAudit
    governance: dict[str, Any]
    verdict: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Outcome:
    target_draw_number: int
    target_draw_date: str
    champion_mean_matches: float
    challenger_mean_matches: float
    random_mean_matches: float
    champion_total_matches: int
    challenger_total_matches: int
    random_total_matches: int
    champion_prize_rate: float
    challenger_prize_rate: float
    random_prize_rate: float


def run_stage22_pair_network_evaluation(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = 3,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    ml_refit_interval: int = DEFAULT_ML_REFIT_INTERVAL,
    experiment_ledger_path: str | Path = DEFAULT_EXPERIMENT_LEDGER_PATH,
    preregistration_path: str | Path = "data/exports/v2_stage22_pair_network_preregistration.json",
) -> Stage22Result:
    if str(lottery.code) != "MINI_LOTO":
        raise ResearchValidationError("Stage 22 pair-network evaluation supports MINI_LOTO only")
    if tickets_per_draw <= 0:
        raise ResearchValidationError("tickets_per_draw must be positive")
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    ordered_input = validate_lottery_dataset(draws, lottery)
    ordered = tuple(
        draw for draw in ordered_input if draw.draw_number <= PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW
    )
    excluded = len(ordered_input) - len(ordered)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough Mini Loto history for Stage 22")
    if ordered[-1].draw_number != PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW:
        raise ResearchValidationError(
            "Stage 22 historical dataset must be frozen at Mini Loto #1400"
        )
    preflight = preflight_validate_benchmark_dataset(ordered, lottery)
    preregistration = stage22_preregistration_payload(preflight.dataset_hash)
    save_stage22_preregistration(preregistration, preregistration_path)

    champion_blocks = build_walk_forward_feature_blocks(
        ordered, lottery, FEATURE_GROUPS["pair_only"]
    )
    challenger_blocks = build_pair_network_feature_blocks(ordered, lottery)
    outcomes = _walk_forward_pair_network_outcomes(
        ordered,
        lottery,
        champion_blocks,
        challenger_blocks,
        seed=seed,
        tickets_per_draw=tickets_per_draw,
        ml_min_training_draws=ml_min_training_draws,
        ml_refit_interval=ml_refit_interval,
    )
    leakage = run_pair_network_leakage_audit(
        ordered,
        lottery,
        seed=seed,
        ml_min_training_draws=ml_min_training_draws,
    )
    if not leakage.lookahead_safe:
        raise ResearchValidationError("Stage 22 pair-network leakage audit failed")

    challenger_vs_champion = _comparison(
        tuple(
            outcome.challenger_mean_matches - outcome.champion_mean_matches for outcome in outcomes
        ),
        tuple(outcome.champion_mean_matches for outcome in outcomes),
        seed=seed,
        label="stage22-challenger-vs-champion",
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    challenger_vs_random = _comparison(
        tuple(
            outcome.challenger_mean_matches - outcome.random_mean_matches for outcome in outcomes
        ),
        tuple(outcome.random_mean_matches for outcome in outcomes),
        seed=seed,
        label="stage22-challenger-vs-random",
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
    )
    periods = _period_stability(outcomes)
    stable_positive = sum(period.mean_match_difference > 0 for period in periods)
    raw_stage_p_values = {
        "pair_network_v1_vs_champion": challenger_vs_champion.raw_p_value,
        "pair_network_v1_vs_random": challenger_vs_random.raw_p_value,
    }
    stage_adjusted = holm_adjust_p_values(raw_stage_p_values)
    ledger = register_stage22_experiment(
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        raw_p_value=challenger_vs_champion.raw_p_value,
        stage_adjusted_p_value=stage_adjusted["pair_network_v1_vs_champion"],
        ledger_path=experiment_ledger_path,
        seed=seed,
    )
    ledger_entry = _ledger_entry(ledger, stage22_experiment_id(preflight.dataset_hash, seed))
    ledger_adjusted = float(ledger_entry["ledger_adjusted_p_value"])
    bh_adjusted = float(ledger_entry["bh_exploratory_p_value"])
    conclusion = classify_conclusion(
        adjusted_p_value=ledger_adjusted,
        difference_ci=challenger_vs_champion.difference_ci,
        standardized_effect=challenger_vs_champion.effect_size.standardized_mean_difference,
        stable_positive_periods=stable_positive,
        total_periods=len(periods),
    )
    verdict = _verdict(
        challenger_vs_champion=challenger_vs_champion,
        challenger_vs_random=challenger_vs_random,
        champion_ledger_p=0.34136586341365865,
        ledger_adjusted_p_value=ledger_adjusted,
        periods=periods,
        leakage=leakage,
    )
    ledger = register_stage22_experiment(
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        raw_p_value=challenger_vs_champion.raw_p_value,
        stage_adjusted_p_value=stage_adjusted["pair_network_v1_vs_champion"],
        ledger_path=experiment_ledger_path,
        seed=seed,
        conclusion=conclusion,
        status=verdict,
    )
    ledger_entry = _ledger_entry(ledger, stage22_experiment_id(preflight.dataset_hash, seed))
    ledger_adjusted = float(ledger_entry["ledger_adjusted_p_value"])
    bh_adjusted = float(ledger_entry["bh_exploratory_p_value"])
    return Stage22Result(
        schema_version=STAGE22_SCHEMA_VERSION,
        lottery=str(lottery.code),
        dataset_hash=preflight.dataset_hash,
        dataset_range={
            "first_draw_number": preflight.first_draw_number,
            "last_draw_number": preflight.last_draw_number,
            "first_draw_date": preflight.first_draw_date,
            "last_draw_date": preflight.last_draw_date,
            "draw_count": preflight.draw_count,
            "excluded_after_cutoff": excluded,
        },
        preregistration=preregistration,
        pair_only_baseline_features=FEATURE_GROUPS["pair_only"],
        pair_network_features=PAIR_NETWORK_FEATURE_NAMES,
        new_pair_network_features=PAIR_NETWORK_NEW_FEATURES,
        model="logistic_regression",
        model_parameters=_make_model("logistic_regression", seed).get_params(),
        portfolio_method="top_ranked",
        configuration={
            "seed": seed,
            "tickets_per_draw": tickets_per_draw,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": confidence_level,
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": ml_refit_interval,
            "historical_cutoff_draw": PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW,
            "recent_pair_window": PAIR_NETWORK_RECENT_WINDOW,
        },
        sklearn_version=sklearn.__version__,
        champion={
            "model": "logistic_regression",
            "feature_group": "pair_only",
            "mean_matches": mean(tuple(outcome.champion_mean_matches for outcome in outcomes)),
            "sample_size": len(outcomes),
        },
        challenger={
            "model": "logistic_regression",
            "feature_group": PAIR_NETWORK_FEATURE_GROUP,
            "mean_matches": mean(tuple(outcome.challenger_mean_matches for outcome in outcomes)),
            "sample_size": len(outcomes),
        },
        challenger_vs_champion=challenger_vs_champion,
        challenger_vs_random=challenger_vs_random,
        win_tie_loss=_win_tie_loss(outcomes),
        period_stability=periods,
        leakage=leakage,
        governance={
            "experiment_id": stage22_experiment_id(preflight.dataset_hash, seed),
            "mini_loto_hypothesis_count": sum(
                entry.get("lottery") == "MINI_LOTO" for entry in ledger.get("entries", ())
            ),
            "stage_adjusted_p_value": stage_adjusted["pair_network_v1_vs_champion"],
            "ledger_adjusted_p_value": ledger_adjusted,
            "bh_exploratory_p_value": bh_adjusted,
            "unified_conclusion": conclusion,
            "multiple_comparison_method": "Holm; BH exploratory reported separately",
        },
        verdict=verdict,
        warnings=(
            "Stage 22 is historical research only and does not activate a shadow challenger.",
            "Mini Loto #1401 is excluded by the frozen #1400 historical cutoff.",
            "Pair-network features are hypotheses, not winning probabilities.",
        ),
    )


def build_pair_network(
    draws: tuple[HistoricalDraw, ...], lottery: LotteryDefinition
) -> PairNetwork:
    validate_lottery_dataset(draws, lottery)
    pair_counts: Counter[tuple[int, int]] = Counter()
    recent_pair_counts: Counter[tuple[int, int]] = Counter()
    for draw in draws:
        pair_counts.update(combinations(draw.main_numbers, 2))
    for draw in draws[-PAIR_NETWORK_RECENT_WINDOW:]:
        recent_pair_counts.update(combinations(draw.main_numbers, 2))
    return PairNetwork(len(draws), pair_counts, recent_pair_counts)


def pair_network_feature_values(
    network: PairNetwork,
    lottery: LotteryDefinition,
    number: int,
) -> dict[str, float]:
    incident = tuple(weight for pair, weight in network.pair_counts.items() if number in pair)
    recent_incident = tuple(
        weight for pair, weight in network.recent_pair_counts.items() if number in pair
    )
    weighted_degree = float(sum(incident))
    scale = network.draw_count * max(1, lottery.numbers_per_ticket - 1)
    return {
        "weighted_degree": weighted_degree,
        "normalized_weighted_degree": 0.0 if scale == 0 else weighted_degree / scale,
        "neighbor_strength_mean": mean(incident) if incident else 0.0,
        "neighbor_strength_max": float(max(incident) if incident else 0),
        "recent_pair_strength": float(sum(recent_incident)),
    }


def build_pair_network_feature_blocks(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> tuple[DrawFeatureBlock, ...]:
    prior_draws: list[HistoricalDraw] = []
    blocks: list[DrawFeatureBlock] = []
    base_blocks = build_walk_forward_feature_blocks(draws, lottery, FEATURE_GROUPS["pair_only"])
    for draw_index, draw in enumerate(draws):
        network = build_pair_network(tuple(prior_draws), lottery)
        rows = []
        for base_row in base_blocks[draw_index].rows:
            network_values = pair_network_feature_values(network, lottery, base_row.number)
            rows.append(
                NumberFeatureRow(
                    draw_index=base_row.draw_index,
                    draw_number=base_row.draw_number,
                    draw_date=base_row.draw_date,
                    number=base_row.number,
                    features=(
                        *base_row.features,
                        *(network_values[name] for name in PAIR_NETWORK_NEW_FEATURES),
                    ),
                    label=base_row.label,
                )
            )
        blocks.append(
            DrawFeatureBlock(
                draw_index=draw_index,
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                rows=tuple(rows),
            )
        )
        prior_draws.append(draw)
    return tuple(blocks)


def run_pair_network_leakage_audit(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    ml_min_training_draws: int,
) -> LeakageAudit:
    target_index = min(max(ml_min_training_draws, 2), len(draws) - 2)
    original_blocks = build_pair_network_feature_blocks(draws, lottery)
    target_block = original_blocks[target_index]
    target_mutation = list(draws)
    target = target_mutation[target_index]
    target_mutation[target_index] = HistoricalDraw(
        target.lottery,
        target.draw_number,
        target.draw_date,
        tuple(reversed(target.main_numbers)),
        target.bonus_numbers,
    )
    mutated_target_block = build_pair_network_feature_blocks(tuple(target_mutation), lottery)[
        target_index
    ]
    future_mutation = list(draws)
    future = future_mutation[target_index + 1]
    future_mutation[target_index + 1] = HistoricalDraw(
        future.lottery,
        future.draw_number,
        future.draw_date,
        tuple(reversed(future.main_numbers)),
        future.bonus_numbers,
    )
    original_scores = _predict_scores(original_blocks, target_index, seed)
    future_scores = _predict_scores(
        build_pair_network_feature_blocks(tuple(future_mutation), lottery),
        target_index,
        seed,
    )
    training_dates_ok = all(
        training_date < draws[target_index].draw_date.isoformat()
        for training_date in build_training_dataset(original_blocks, target_index)[2]
    )
    target_features_changed = tuple(row.features for row in target_block.rows) != tuple(
        row.features for row in mutated_target_block.rows
    )
    future_prediction_changed = original_scores != future_scores
    return LeakageAudit(
        lookahead_safe=training_dates_ok
        and not target_features_changed
        and not future_prediction_changed,
        training_dates_strictly_before_target=training_dates_ok,
        target_mutation_changes_features=target_features_changed,
        future_mutation_changes_prediction=future_prediction_changed,
    )


def stage22_preregistration_payload(dataset_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "v2-stage22-preregistration-v1",
        "lottery": "MINI_LOTO",
        "champion": "logistic_regression + pair_only + top_ranked",
        "challenger": "logistic_regression + pair_network_v1 + top_ranked",
        "primary_hypothesis": (
            "Pair-network structural information improves prospective number ranking "
            "relative to the existing pair_only champion."
        ),
        "primary_comparison": "challenger_vs_champion",
        "secondary_comparison": "challenger_vs_matched_random",
        "primary_metric": "average main-number matches per ticket in walk-forward evaluation",
        "historical_cutoff": {
            "last_included_draw": PAIR_NETWORK_HISTORICAL_CUTOFF_DRAW,
            "excluded_draws": "draws after #1400, including #1401 if present",
        },
        "dataset_hash": dataset_hash,
        "feature_group": PAIR_NETWORK_FEATURE_GROUP,
        "feature_names": PAIR_NETWORK_FEATURE_NAMES,
    }


def save_stage22_preregistration(
    payload: dict[str, Any],
    output_path: str | Path = "data/exports/v2_stage22_pair_network_preregistration.json",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return path


def save_stage22_result(result: Stage22Result, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(result), encoding="utf-8")
    return path


def save_stage22_summary(
    result: Stage22Result,
    output_path: str | Path = "data/exports/v2_stage22_pair_network_summary.json",
) -> Path:
    payload = {
        "schema_version": "v2-stage22-pair-network-summary-v1",
        "lottery": result.lottery,
        "dataset_hash": result.dataset_hash,
        "champion_mean": result.champion["mean_matches"],
        "challenger_mean": result.challenger["mean_matches"],
        "diff_vs_champion": result.challenger_vs_champion.difference,
        "diff_vs_random": result.challenger_vs_random.difference,
        "ledger_adjusted_p_value": result.governance["ledger_adjusted_p_value"],
        "verdict": result.verdict,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return path


def register_stage22_experiment(
    *,
    lottery: str,
    dataset_hash: str,
    raw_p_value: float,
    stage_adjusted_p_value: float,
    ledger_path: str | Path,
    seed: int = DEFAULT_STAGE05_SEED,
    conclusion: str = "pending",
    status: str = "pending",
) -> dict[str, Any]:
    path = Path(ledger_path)
    ledger = _load_ledger(path)
    experiment_id = stage22_experiment_id(dataset_hash, seed)
    entry = {
        "experiment_id": experiment_id,
        "stage": "22",
        "lottery": lottery,
        "dataset_hash": dataset_hash,
        "hypothesis": "Mini Loto pair_network_v1 improves over pair_only champion",
        "model": "logistic_regression",
        "feature_group": PAIR_NETWORK_FEATURE_GROUP,
        "comparison": "pair_network_v1_vs_pair_only_champion",
        "raw_p_value": raw_p_value,
        "stage_adjusted_p_value": stage_adjusted_p_value,
        "ledger_adjusted_p_value": stage_adjusted_p_value,
        "bh_exploratory_p_value": raw_p_value,
        "conclusion": conclusion,
        "status": status,
    }
    entries_by_id = {
        str(item["experiment_id"]): item
        for item in ledger.get("entries", ())
        if "experiment_id" in item
    }
    entries_by_id[experiment_id] = entry
    entries = sorted(entries_by_id.values(), key=lambda item: str(item["experiment_id"]))
    p_values = {str(item["experiment_id"]): float(item["raw_p_value"]) for item in entries}
    holm = holm_adjust_p_values(p_values)
    bh = benjamini_hochberg_adjust_p_values(p_values)
    for item in entries:
        item["ledger_adjusted_p_value"] = holm[str(item["experiment_id"])]
        item["bh_exploratory_p_value"] = bh[str(item["experiment_id"])]
    payload = {
        "schema_version": "v2-experiment-ledger-v1",
        "multiple_comparison_method": "holm",
        "exploratory_method": "benjamini_hochberg",
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return payload


def stage22_experiment_id(dataset_hash: str, seed: int) -> str:
    payload = "|".join(
        (
            STAGE22_SCHEMA_VERSION,
            "MINI_LOTO",
            dataset_hash,
            PAIR_NETWORK_FEATURE_GROUP,
            str(seed),
        )
    )
    return "V2-EXP-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _walk_forward_pair_network_outcomes(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    champion_blocks: tuple[DrawFeatureBlock, ...],
    challenger_blocks: tuple[DrawFeatureBlock, ...],
    *,
    seed: int,
    tickets_per_draw: int,
    ml_min_training_draws: int,
    ml_refit_interval: int,
) -> tuple[_Outcome, ...]:
    champion_model = None
    challenger_model = None
    champion_fit_index: int | None = None
    challenger_fit_index: int | None = None
    outcomes: list[_Outcome] = []
    for target_index in range(ml_min_training_draws, len(draws)):
        target = draws[target_index]
        if (
            champion_model is None
            or champion_fit_index is None
            or target_index - champion_fit_index >= ml_refit_interval
        ):
            x_train, y_train, dates = build_training_dataset(champion_blocks, target_index)
            _assert_training_dates(dates, target)
            champion_model = _make_model("logistic_regression", seed)
            champion_model.fit(x_train, y_train)
            champion_fit_index = target_index
        if (
            challenger_model is None
            or challenger_fit_index is None
            or target_index - challenger_fit_index >= ml_refit_interval
        ):
            x_train, y_train, dates = build_training_dataset(challenger_blocks, target_index)
            _assert_training_dates(dates, target)
            challenger_model = _make_model("logistic_regression", seed)
            challenger_model.fit(x_train, y_train)
            challenger_fit_index = target_index
        champion_scores = _scores_from_fitted_model(champion_model, champion_blocks[target_index])
        challenger_scores = _scores_from_fitted_model(
            challenger_model, challenger_blocks[target_index]
        )
        champion_tickets = _tickets(champion_scores, lottery, tickets_per_draw)
        challenger_tickets = _tickets(challenger_scores, lottery, tickets_per_draw)
        random_tickets = generate_distinct_random_tickets(
            lottery,
            random.Random(seed + target.draw_number),
            tickets_per_draw,
        )
        outcomes.append(
            _outcome(
                target,
                lottery,
                champion_tickets=champion_tickets,
                challenger_tickets=challenger_tickets,
                random_tickets=random_tickets,
            )
        )
    return tuple(outcomes)


def _tickets(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    tickets_per_draw: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        ticket.numbers
        for ticket in _generate_ranked_tickets(
            scores,
            lottery,
            tickets_per_draw=tickets_per_draw,
            method="top_ranked",
            candidate_pool_size=50,
        )
    )


def _outcome(
    target: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    champion_tickets: tuple[tuple[int, ...], ...],
    challenger_tickets: tuple[tuple[int, ...], ...],
    random_tickets: tuple[tuple[int, ...], ...],
) -> _Outcome:
    champion = tuple(match_ticket(ticket, target, lottery) for ticket in champion_tickets)
    challenger = tuple(match_ticket(ticket, target, lottery) for ticket in challenger_tickets)
    random_matches = tuple(match_ticket(ticket, target, lottery) for ticket in random_tickets)
    champion_counts = tuple(result.main_match_count for result in champion)
    challenger_counts = tuple(result.main_match_count for result in challenger)
    random_counts = tuple(result.main_match_count for result in random_matches)
    return _Outcome(
        target_draw_number=target.draw_number,
        target_draw_date=target.draw_date.isoformat(),
        champion_mean_matches=mean(champion_counts),
        challenger_mean_matches=mean(challenger_counts),
        random_mean_matches=mean(random_counts),
        champion_total_matches=sum(champion_counts),
        challenger_total_matches=sum(challenger_counts),
        random_total_matches=sum(random_counts),
        champion_prize_rate=sum(result.qualifies_for_prize for result in champion) / len(champion),
        challenger_prize_rate=sum(result.qualifies_for_prize for result in challenger)
        / len(challenger),
        random_prize_rate=sum(result.qualifies_for_prize for result in random_matches)
        / len(random_matches),
    )


def _comparison(
    differences: tuple[float, ...],
    baseline_values: tuple[float, ...],
    *,
    seed: int,
    label: str,
    bootstrap_replications: int,
    confidence_level: float,
) -> Stage22Comparison:
    diff = mean(differences) if differences else 0.0
    baseline = mean(baseline_values) if baseline_values else 0.0
    return Stage22Comparison(
        challenger_value=baseline + diff,
        baseline_value=baseline,
        difference=diff,
        difference_ci=bootstrap_confidence_interval(
            differences,
            seed=_derived_seed(seed, f"{label}-ci"),
            replications=bootstrap_replications,
            confidence_level=confidence_level,
        ),
        effect_size=_effect_size(differences, baseline),
        raw_p_value=paired_permutation_p_value(
            differences,
            seed=_derived_seed(seed, f"{label}-permutation"),
            replications=bootstrap_replications,
        ),
    )


def _period_stability(outcomes: tuple[_Outcome, ...]) -> tuple[PeriodStability, ...]:
    periods = (
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2023", "2020-01-01", "2023-12-31"),
        ("2024-latest", "2024-01-01", "9999-12-31"),
    )
    rows: list[PeriodStability] = []
    for label, start, end in periods:
        selected = tuple(
            outcome for outcome in outcomes if start <= outcome.target_draw_date <= end
        )
        if not selected:
            continue
        diff = mean(
            outcome.challenger_mean_matches - outcome.champion_mean_matches for outcome in selected
        )
        prize_diff = mean(
            outcome.challenger_prize_rate - outcome.champion_prize_rate for outcome in selected
        )
        rows.append(
            PeriodStability(
                period=label,
                target_draws=len(selected),
                mean_match_difference=diff,
                prize_rate_difference=prize_diff,
                direction="positive" if diff > 0 else "negative" if diff < 0 else "zero",
            )
        )
    return tuple(rows)


def _win_tie_loss(outcomes: tuple[_Outcome, ...]) -> WinTieLoss:
    wins = sum(
        outcome.challenger_total_matches > outcome.champion_total_matches for outcome in outcomes
    )
    ties = sum(
        outcome.challenger_total_matches == outcome.champion_total_matches for outcome in outcomes
    )
    losses = len(outcomes) - wins - ties
    return WinTieLoss(wins=wins, ties=ties, losses=losses)


def _verdict(
    *,
    challenger_vs_champion: Stage22Comparison,
    challenger_vs_random: Stage22Comparison,
    champion_ledger_p: float,
    ledger_adjusted_p_value: float,
    periods: tuple[PeriodStability, ...],
    leakage: LeakageAudit,
) -> str:
    if not leakage.lookahead_safe:
        return VERDICT_RETIRE
    stable_positive = sum(period.mean_match_difference > 0 for period in periods)
    driven_by_one_period = stable_positive < max(1, len(periods) - 1)
    if (
        challenger_vs_champion.difference > 0
        and challenger_vs_champion.effect_size.absolute_difference > 0
        and challenger_vs_random.difference > 0
        and challenger_vs_champion.difference_ci.upper >= 0
        and ledger_adjusted_p_value < champion_ledger_p
        and not driven_by_one_period
    ):
        return VERDICT_KEEP
    if challenger_vs_champion.difference < 0 and challenger_vs_random.difference < 0:
        return VERDICT_RETIRE
    return VERDICT_INCONCLUSIVE


def _predict_scores(
    blocks: tuple[DrawFeatureBlock, ...],
    target_index: int,
    seed: int,
) -> dict[int, float]:
    x_train, y_train, _dates = build_training_dataset(blocks, target_index)
    model = _make_model("logistic_regression", seed)
    model.fit(x_train, y_train)
    return _scores_from_fitted_model(model, blocks[target_index])


def _assert_training_dates(dates: tuple[str, ...], target: HistoricalDraw) -> None:
    if not all(date_text < target.draw_date.isoformat() for date_text in dates):
        raise ResearchValidationError("training rows include target or future draw")


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    std = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=0.0 if std == 0 else absolute / std,
    )


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "v2-experiment-ledger-v1",
            "multiple_comparison_method": "holm",
            "exploratory_method": "benjamini_hochberg",
            "entries": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_entry(ledger: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for entry in ledger.get("entries", ()):
        if entry.get("experiment_id") == experiment_id:
            return entry
    raise ResearchValidationError("Stage 22 experiment was not registered in ledger")
