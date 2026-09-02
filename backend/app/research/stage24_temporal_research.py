from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.baseline_benchmark import DEFAULT_STAGE05_SEED
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.persistence import research_result_json
from backend.app.research.statistical_evaluation import (
    DEFAULT_CONFIDENCE_LEVEL,
    ConfidenceInterval,
    EffectSize,
    bootstrap_confidence_interval,
    holm_adjust_p_values,
    paired_permutation_p_value,
)

STAGE24_SCHEMA_VERSION = "v2-stage24-temporal-discovery-v1"
STAGE24_DECISION_SCHEMA_VERSION = "v2-stage24-frozen-decision-v1"
STAGE24_DISCOVERY_CUTOFF_DRAW = 1401
STAGE24_HOLDOUT_DRAW = 1402
STAGE24_TOP_K = 5
STAGE24_MIN_TRAINING_DRAWS = 100
STAGE24_BOOTSTRAP_REPLICATIONS = 10_000
STAGE24_MONTE_CARLO_REPLICATIONS = 10_000
STAGE24_OUTPUT_DIR = Path("data") / "exports" / "stage24"

TEMPORAL_SIGNALS = (
    "recent_frequency_momentum",
    "recent_frequency_mean_reversion",
    "consecutive_draw_carryover",
    "multi_lag_recurrence_t2",
    "transition_from_previous_draw",
    "number_persistence",
    "short_window_regime_concentration",
    "hot_cold_state_change",
    "bonus_to_main_follow",
)


@dataclass(frozen=True, slots=True)
class TemporalSignalObservation:
    draw_number: int
    draw_date: str
    signal_matches: int
    random_matches: int
    selected_numbers: tuple[int, ...]
    random_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalSignalResult:
    signal: str
    sample_size: int
    top_k: int
    mean_matches: float
    random_mean_matches: float
    difference: float
    difference_ci: ConfidenceInterval
    raw_p_value: float
    adjusted_p_value: float
    bh_exploratory_p_value: float
    effect_size: EffectSize
    classification: str
    observations: tuple[TemporalSignalObservation, ...]


@dataclass(frozen=True, slots=True)
class ConcentrationCheck:
    draw_numbers: tuple[int, ...]
    unique_values: int
    maximum_repeat_count: int
    repeated_slots: int
    concentration_index: int
    rolling_window_count: int
    rolling_repeated_slots_percentile: float
    rolling_concentration_percentile: float
    monte_carlo_repeated_slots_percentile: float
    monte_carlo_concentration_percentile: float


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    draw_number: int | None
    draw_date: str | None
    evaluated: bool
    signal: str
    selected_numbers: tuple[int, ...]
    actual_main_numbers: tuple[int, ...]
    matches: int
    expected_random_matches: float
    discovery_signal_mean_matches: float
    interpretation: str


@dataclass(frozen=True, slots=True)
class Stage24TemporalResearchResult:
    schema_version: str
    lottery: str
    discovery_cutoff_draw: int
    discovery_dataset_hash: str
    discovery_draw_count: int
    discovery_range: dict[str, str | int]
    configuration: dict[str, Any]
    signals: dict[str, TemporalSignalResult]
    concentration_check: ConcentrationCheck
    frozen_decision: dict[str, Any]
    frozen_decision_hash: str
    holdout: HoldoutResult
    warnings: tuple[str, ...]


def run_stage24_temporal_research(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    seed: int = DEFAULT_STAGE05_SEED,
    min_training_draws: int = STAGE24_MIN_TRAINING_DRAWS,
    bootstrap_replications: int = STAGE24_BOOTSTRAP_REPLICATIONS,
    monte_carlo_replications: int = STAGE24_MONTE_CARLO_REPLICATIONS,
    output_dir: str | Path | None = None,
) -> Stage24TemporalResearchResult:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 24 temporal research supports MINI_LOTO only")
    if min_training_draws <= 0:
        raise ResearchValidationError("min_training_draws must be positive")
    if bootstrap_replications <= 0:
        raise ResearchValidationError("bootstrap_replications must be positive")
    if monte_carlo_replications <= 0:
        raise ResearchValidationError("monte_carlo_replications must be positive")

    ordered_input = validate_lottery_dataset(draws, lottery)
    discovery = discovery_slice(ordered_input, lottery)
    if len(discovery) <= min_training_draws:
        raise ResearchValidationError("not enough Mini Loto history for Stage 24")
    discovery_hash = calculate_dataset_hash(discovery)
    signal_results = evaluate_temporal_signals(
        discovery,
        lottery,
        seed=seed,
        min_training_draws=min_training_draws,
        bootstrap_replications=bootstrap_replications,
    )
    concentration = recent_window_concentration_check(
        discovery,
        lottery,
        seed=seed,
        monte_carlo_replications=monte_carlo_replications,
    )
    decision = frozen_decision_payload(
        discovery,
        signal_results,
        concentration,
        seed=seed,
        min_training_draws=min_training_draws,
        bootstrap_replications=bootstrap_replications,
        monte_carlo_replications=monte_carlo_replications,
    )
    decision_hash = stable_payload_hash(decision)
    frozen_decision = {**decision, "decision_hash": decision_hash}
    if output_dir is not None:
        save_stage24_frozen_decision(frozen_decision, Path(output_dir))
    holdout = evaluate_holdout_after_frozen_decision(
        ordered_input,
        lottery,
        frozen_decision=frozen_decision,
    )
    result = Stage24TemporalResearchResult(
        schema_version=STAGE24_SCHEMA_VERSION,
        lottery=str(lottery.code),
        discovery_cutoff_draw=STAGE24_DISCOVERY_CUTOFF_DRAW,
        discovery_dataset_hash=discovery_hash,
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
            "top_k": STAGE24_TOP_K,
            "min_training_draws": min_training_draws,
            "bootstrap_replications": bootstrap_replications,
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "monte_carlo_replications": monte_carlo_replications,
            "signals": TEMPORAL_SIGNALS,
            "discovery_cutoff_draw": STAGE24_DISCOVERY_CUTOFF_DRAW,
            "holdout_draw": STAGE24_HOLDOUT_DRAW,
        },
        signals=signal_results,
        concentration_check=concentration,
        frozen_decision=frozen_decision,
        frozen_decision_hash=decision_hash,
        holdout=holdout,
        warnings=(
            "Stage 24 is research-only and does not change production strategy or predictions.",
            "Discovery is frozen at Mini Loto #1401; #1402 is holdout-only.",
            "Draws after #1402 are excluded from the Stage 24 decision and holdout.",
            "Temporal signals are hypotheses, not winning probabilities.",
        ),
    )
    if output_dir is not None:
        save_stage24_outputs(result, output_dir)
    return result


def discovery_slice(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 24 temporal research supports MINI_LOTO only")
    ordered = validate_lottery_dataset(draws, lottery)
    sliced = tuple(draw for draw in ordered if draw.draw_number <= STAGE24_DISCOVERY_CUTOFF_DRAW)
    if not sliced or sliced[-1].draw_number != STAGE24_DISCOVERY_CUTOFF_DRAW:
        raise ResearchValidationError("Stage 24 discovery requires Mini Loto history through #1401")
    return sliced


def evaluate_temporal_signals(
    discovery_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    min_training_draws: int,
    bootstrap_replications: int,
) -> dict[str, TemporalSignalResult]:
    observations_by_signal = {
        signal: _signal_observations(
            signal,
            discovery_draws,
            lottery,
            seed=seed,
            min_training_draws=min_training_draws,
        )
        for signal in TEMPORAL_SIGNALS
    }
    raw_p_values = {
        signal: paired_permutation_p_value(
            tuple(obs.signal_matches - obs.random_matches for obs in observations),
            seed=_derived_seed(seed, f"{signal}-permutation"),
            replications=bootstrap_replications,
        )
        for signal, observations in observations_by_signal.items()
    }
    adjusted = holm_adjust_p_values(raw_p_values)
    exploratory = benjamini_hochberg_adjust_p_values(raw_p_values)
    return {
        signal: _signal_result(
            signal,
            observations,
            adjusted_p_value=adjusted[signal],
            bh_exploratory_p_value=exploratory[signal],
            raw_p_value=raw_p_values[signal],
            seed=seed,
            bootstrap_replications=bootstrap_replications,
        )
        for signal, observations in observations_by_signal.items()
    }


def score_temporal_signal(
    signal: str,
    history: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> dict[int, float]:
    if signal not in TEMPORAL_SIGNALS:
        raise ResearchValidationError(f"unknown Stage 24 temporal signal: {signal}")
    numbers = range(lottery.number_min, lottery.number_max + 1)
    recent4 = _main_counts(history[-4:])
    recent20 = _main_counts(history[-20:])
    previous4 = _main_counts(history[-8:-4])
    all_counts = _main_counts(history)
    gaps = _current_gaps(history, lottery)
    transition = _transition_scores(history, lottery)
    persistence = _persistence_scores(history, lottery)
    bonus_follow = _bonus_follow_scores(history, lottery)
    last_main = set(history[-1].main_numbers) if history else set()
    lag2_main = set(history[-2].main_numbers) if len(history) >= 2 else set()
    lag3_main = set(history[-3].main_numbers) if len(history) >= 3 else set()
    lag4_main = set(history[-4].main_numbers) if len(history) >= 4 else set()

    scores: dict[int, float] = {}
    for number in numbers:
        if signal == "recent_frequency_momentum":
            scores[number] = recent4[number] / 4 - recent20[number] / 20
        elif signal == "recent_frequency_mean_reversion":
            historical_rate = all_counts[number] / max(1, len(history))
            scores[number] = historical_rate - recent4[number] / 4 + gaps[number] / len(history)
        elif signal == "consecutive_draw_carryover":
            scores[number] = 1.0 if number in last_main else 0.0
        elif signal == "multi_lag_recurrence_t2":
            scores[number] = (
                (1.0 if number in lag2_main else 0.0)
                + (0.5 if number in lag3_main else 0.0)
                + (0.25 if number in lag4_main else 0.0)
            )
        elif signal == "transition_from_previous_draw":
            scores[number] = transition[number]
        elif signal == "number_persistence":
            scores[number] = persistence[number] if number in last_main else 0.0
        elif signal == "short_window_regime_concentration":
            scores[number] = float(recent4[number])
        elif signal == "hot_cold_state_change":
            scores[number] = float(recent4[number] - previous4[number])
        elif signal == "bonus_to_main_follow":
            scores[number] = bonus_follow[number]
    return scores


def select_top_numbers(scores: dict[int, float], top_k: int = STAGE24_TOP_K) -> tuple[int, ...]:
    return tuple(
        number for number, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    )


def frozen_decision_payload(
    discovery_draws: tuple[HistoricalDraw, ...],
    signal_results: dict[str, TemporalSignalResult],
    concentration: ConcentrationCheck,
    *,
    seed: int,
    min_training_draws: int,
    bootstrap_replications: int,
    monte_carlo_replications: int,
) -> dict[str, Any]:
    strongest = strongest_signal(signal_results)
    discovery_hash = calculate_dataset_hash(discovery_draws)
    recommendation = stage25_recommendation(strongest)
    return {
        "schema_version": STAGE24_DECISION_SCHEMA_VERSION,
        "lottery": "MINI_LOTO",
        "discovery_cutoff_draw": STAGE24_DISCOVERY_CUTOFF_DRAW,
        "discovery_dataset_hash": discovery_hash,
        "discovery_draw_count": len(discovery_draws),
        "discovery_last_draw_date": discovery_draws[-1].draw_date.isoformat(),
        "strongest_signal": strongest.signal,
        "strongest_signal_mean": strongest.mean_matches,
        "strongest_signal_random_mean": strongest.random_mean_matches,
        "strongest_signal_difference": strongest.difference,
        "strongest_signal_raw_p_value": strongest.raw_p_value,
        "strongest_signal_adjusted_p_value": strongest.adjusted_p_value,
        "strongest_signal_classification": strongest.classification,
        "stage25_challenger_justified": recommendation["justified"],
        "stage25_recommendation": recommendation["recommendation"],
        "configuration": {
            "seed": seed,
            "top_k": STAGE24_TOP_K,
            "min_training_draws": min_training_draws,
            "bootstrap_replications": bootstrap_replications,
            "monte_carlo_replications": monte_carlo_replications,
            "signals": TEMPORAL_SIGNALS,
            "multiplicity_correction": "Holm across Stage 24 temporal signals",
            "bh_exploratory_reported": True,
        },
        "recent_concentration": {
            "draw_numbers": concentration.draw_numbers,
            "unique_values": concentration.unique_values,
            "maximum_repeat_count": concentration.maximum_repeat_count,
            "repeated_slots": concentration.repeated_slots,
            "rolling_repeated_slots_percentile": concentration.rolling_repeated_slots_percentile,
            "monte_carlo_repeated_slots_percentile": (
                concentration.monte_carlo_repeated_slots_percentile
            ),
        },
        "frozen_before_holdout": True,
        "excluded_from_discovery": {
            "holdout_draw": STAGE24_HOLDOUT_DRAW,
            "later_draws": "all draws after #1402",
        },
    }


def strongest_signal(signal_results: dict[str, TemporalSignalResult]) -> TemporalSignalResult:
    return sorted(
        signal_results.values(),
        key=lambda result: (
            -result.difference,
            result.raw_p_value,
            result.signal,
        ),
    )[0]


def stage25_recommendation(strongest: TemporalSignalResult) -> dict[str, Any]:
    justified = (
        strongest.difference > 0
        and strongest.difference_ci.lower > 0
        and strongest.adjusted_p_value < 0.05
        and strongest.classification == "EVIDENCE"
    )
    return {
        "justified": justified,
        "recommendation": (
            "Stage 25 challenger justified for shadow-only preregistration"
            if justified
            else "No Stage 25 temporal challenger justified from Stage 24 discovery"
        ),
    }


def evaluate_holdout_after_frozen_decision(
    all_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    frozen_decision: dict[str, Any],
) -> HoldoutResult:
    if not frozen_decision.get("frozen_before_holdout"):
        raise ResearchValidationError("Stage 24 decision must be frozen before holdout evaluation")
    discovery = discovery_slice(all_draws, lottery)
    holdout = next((draw for draw in all_draws if draw.draw_number == STAGE24_HOLDOUT_DRAW), None)
    signal = str(frozen_decision["strongest_signal"])
    scores = score_temporal_signal(signal, discovery, lottery)
    selected = select_top_numbers(scores)
    if holdout is None:
        return HoldoutResult(
            draw_number=None,
            draw_date=None,
            evaluated=False,
            signal=signal,
            selected_numbers=selected,
            actual_main_numbers=(),
            matches=0,
            expected_random_matches=STAGE24_TOP_K * lottery.numbers_per_ticket / lottery.number_max,
            discovery_signal_mean_matches=float(frozen_decision.get("strongest_signal_mean", 0.0)),
            interpretation="Mini Loto #1402 is not available in the supplied data.",
        )
    matches = len(set(selected) & set(holdout.main_numbers))
    expected = STAGE24_TOP_K * lottery.numbers_per_ticket / lottery.number_max
    return HoldoutResult(
        draw_number=holdout.draw_number,
        draw_date=holdout.draw_date.isoformat(),
        evaluated=True,
        signal=signal,
        selected_numbers=selected,
        actual_main_numbers=holdout.main_numbers,
        matches=matches,
        expected_random_matches=expected,
        discovery_signal_mean_matches=float(frozen_decision.get("strongest_signal_mean", 0.0)),
        interpretation=(
            "One holdout draw is observational only and does not alter the frozen Stage 24 "
            "conclusion."
        ),
    )


def recent_window_concentration_check(
    discovery_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    monte_carlo_replications: int,
) -> ConcentrationCheck:
    recent = discovery_draws[-4:]
    recent_stats = _window_concentration(recent)
    rolling = tuple(
        _window_concentration(discovery_draws[index : index + 4])
        for index in range(0, len(discovery_draws) - 3)
    )
    rng = random.Random(_derived_seed(seed, "stage24-concentration-monte-carlo"))
    monte_carlo = tuple(
        _numbers_concentration(
            tuple(
                number
                for _ in range(4)
                for number in sorted(
                    rng.sample(
                        range(lottery.number_min, lottery.number_max + 1),
                        lottery.numbers_per_ticket,
                    )
                )
            )
        )
        for _ in range(monte_carlo_replications)
    )
    return ConcentrationCheck(
        draw_numbers=tuple(draw.draw_number for draw in recent),
        unique_values=recent_stats["unique_values"],
        maximum_repeat_count=recent_stats["maximum_repeat_count"],
        repeated_slots=recent_stats["repeated_slots"],
        concentration_index=recent_stats["concentration_index"],
        rolling_window_count=len(rolling),
        rolling_repeated_slots_percentile=_percentile_rank(
            tuple(item["repeated_slots"] for item in rolling), recent_stats["repeated_slots"]
        ),
        rolling_concentration_percentile=_percentile_rank(
            tuple(item["concentration_index"] for item in rolling),
            recent_stats["concentration_index"],
        ),
        monte_carlo_repeated_slots_percentile=_percentile_rank(
            tuple(item["repeated_slots"] for item in monte_carlo), recent_stats["repeated_slots"]
        ),
        monte_carlo_concentration_percentile=_percentile_rank(
            tuple(item["concentration_index"] for item in monte_carlo),
            recent_stats["concentration_index"],
        ),
    )


def stable_payload_hash(payload: dict[str, Any]) -> str:
    canonical = research_result_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_stage24_outputs(
    result: Stage24TemporalResearchResult, output_dir: str | Path
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    decision_path = save_stage24_frozen_decision(result.frozen_decision, root)
    result_path = root / "v2_stage24_temporal_research.json"
    summary_path = root / "v2_stage24_summary.json"
    result_path.write_text(research_result_json(result), encoding="utf-8")
    summary_path.write_text(
        research_result_json(
            {
                "schema_version": "v2-stage24-summary-v1",
                "lottery": result.lottery,
                "discovery_cutoff_draw": result.discovery_cutoff_draw,
                "discovery_dataset_hash": result.discovery_dataset_hash,
                "strongest_signal": result.frozen_decision["strongest_signal"],
                "classification": result.frozen_decision["strongest_signal_classification"],
                "adjusted_p_value": result.frozen_decision["strongest_signal_adjusted_p_value"],
                "stage25_challenger_justified": result.frozen_decision[
                    "stage25_challenger_justified"
                ],
                "frozen_decision_hash": result.frozen_decision_hash,
                "holdout_matches": result.holdout.matches if result.holdout.evaluated else None,
            }
        ),
        encoding="utf-8",
    )
    return {
        "decision": str(decision_path),
        "result": str(result_path),
        "summary": str(summary_path),
    }


def save_stage24_frozen_decision(frozen_decision: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "v2_stage24_frozen_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(frozen_decision), encoding="utf-8")
    return path


def _signal_observations(
    signal: str,
    discovery_draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    seed: int,
    min_training_draws: int,
) -> tuple[TemporalSignalObservation, ...]:
    rows: list[TemporalSignalObservation] = []
    for target_index in range(min_training_draws, len(discovery_draws)):
        history = discovery_draws[:target_index]
        target = discovery_draws[target_index]
        selected = select_top_numbers(score_temporal_signal(signal, history, lottery))
        random_numbers = tuple(
            sorted(
                random.Random(_derived_seed(seed, f"{signal}-{target.draw_number}")).sample(
                    range(lottery.number_min, lottery.number_max + 1),
                    STAGE24_TOP_K,
                )
            )
        )
        rows.append(
            TemporalSignalObservation(
                draw_number=target.draw_number,
                draw_date=target.draw_date.isoformat(),
                signal_matches=len(set(selected) & set(target.main_numbers)),
                random_matches=len(set(random_numbers) & set(target.main_numbers)),
                selected_numbers=selected,
                random_numbers=random_numbers,
            )
        )
    return tuple(rows)


def _signal_result(
    signal: str,
    observations: tuple[TemporalSignalObservation, ...],
    *,
    adjusted_p_value: float,
    bh_exploratory_p_value: float,
    raw_p_value: float,
    seed: int,
    bootstrap_replications: int,
) -> TemporalSignalResult:
    signal_matches = tuple(obs.signal_matches for obs in observations)
    random_matches = tuple(obs.random_matches for obs in observations)
    differences = tuple(
        left - right for left, right in zip(signal_matches, random_matches, strict=True)
    )
    difference = mean(differences) if differences else 0.0
    ci = bootstrap_confidence_interval(
        differences,
        seed=_derived_seed(seed, f"{signal}-ci"),
        replications=bootstrap_replications,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    effect = _effect_size(differences, mean(random_matches) if random_matches else 0.0)
    return TemporalSignalResult(
        signal=signal,
        sample_size=len(observations),
        top_k=STAGE24_TOP_K,
        mean_matches=mean(signal_matches) if signal_matches else 0.0,
        random_mean_matches=mean(random_matches) if random_matches else 0.0,
        difference=difference,
        difference_ci=ci,
        raw_p_value=raw_p_value,
        adjusted_p_value=adjusted_p_value,
        bh_exploratory_p_value=bh_exploratory_p_value,
        effect_size=effect,
        classification=_classification(difference, ci, adjusted_p_value),
        observations=observations,
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


def _effect_size(differences: tuple[float, ...], baseline_value: float) -> EffectSize:
    absolute = mean(differences) if differences else 0.0
    std = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=absolute,
        relative_difference=None if baseline_value == 0 else absolute / baseline_value,
        standardized_mean_difference=0.0 if std == 0 else absolute / std,
    )


def _main_counts(draws: tuple[HistoricalDraw, ...]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for draw in draws:
        counts.update(draw.main_numbers)
    return counts


def _current_gaps(draws: tuple[HistoricalDraw, ...], lottery: LotteryDefinition) -> dict[int, int]:
    gaps: dict[int, int] = {}
    for number in range(lottery.number_min, lottery.number_max + 1):
        gap = 0
        for draw in reversed(draws):
            if number in draw.main_numbers:
                break
            gap += 1
        gaps[number] = gap
    return gaps


def _transition_scores(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> dict[int, float]:
    if len(draws) < 2:
        return {number: 0.0 for number in range(lottery.number_min, lottery.number_max + 1)}
    pair_counts: Counter[tuple[int, int]] = Counter()
    previous_counts: Counter[int] = Counter()
    for previous, current in zip(draws, draws[1:], strict=False):
        previous_counts.update(previous.main_numbers)
        for left in previous.main_numbers:
            for right in current.main_numbers:
                pair_counts[(left, right)] += 1
    latest = draws[-1]
    scores = {}
    for number in range(lottery.number_min, lottery.number_max + 1):
        score = 0.0
        for previous_number in latest.main_numbers:
            denominator = max(1, previous_counts[previous_number])
            score += pair_counts[(previous_number, number)] / denominator
        scores[number] = score
    return scores


def _persistence_scores(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> dict[int, float]:
    opportunities: Counter[int] = Counter()
    repeats: Counter[int] = Counter()
    for previous, current in zip(draws, draws[1:], strict=False):
        for number in previous.main_numbers:
            opportunities[number] += 1
            if number in current.main_numbers:
                repeats[number] += 1
    return {
        number: repeats[number] / opportunities[number] if opportunities[number] else 0.0
        for number in range(lottery.number_min, lottery.number_max + 1)
    }


def _bonus_follow_scores(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
) -> dict[int, float]:
    opportunities: Counter[int] = Counter()
    follows: Counter[int] = Counter()
    for previous, current in zip(draws, draws[1:], strict=False):
        for bonus in previous.bonus_numbers:
            opportunities[bonus] += 1
            if bonus in current.main_numbers:
                follows[bonus] += 1
    latest_bonus = set(draws[-1].bonus_numbers) if draws else set()
    return {
        number: (
            (follows[number] / opportunities[number] if opportunities[number] else 0.0)
            + (1.0 if number in latest_bonus else 0.0)
        )
        for number in range(lottery.number_min, lottery.number_max + 1)
    }


def _window_concentration(draws: tuple[HistoricalDraw, ...]) -> dict[str, int]:
    return _numbers_concentration(tuple(number for draw in draws for number in draw.main_numbers))


def _numbers_concentration(numbers: tuple[int, ...]) -> dict[str, int]:
    counts = Counter(numbers)
    return {
        "unique_values": len(counts),
        "maximum_repeat_count": max(counts.values()) if counts else 0,
        "repeated_slots": len(numbers) - len(counts),
        "concentration_index": sum(value * value for value in counts.values()),
    }


def _percentile_rank(values: tuple[int, ...], observed: int) -> float:
    if not values:
        return 0.0
    below = sum(value < observed for value in values)
    equal = sum(value == observed for value in values)
    return (below + 0.5 * equal) / len(values)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return int(digest[:16], 16)
