from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.baseline_benchmark import DEFAULT_STAGE05_SEED
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.ml_baseline import _derived_seed, _NumberFeatureState
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.production import (
    _score_future_numbers,
    next_scheduled_draw_date,
    production_strategy_for_lottery,
)
from backend.app.research.stage25_ranking_discrimination import rank_numbers_from_scores
from backend.app.research.statistical_evaluation import (
    DEFAULT_BOOTSTRAP_REPLICATIONS,
    DEFAULT_CONFIDENCE_LEVEL,
    EffectSize,
    bootstrap_confidence_interval,
    paired_permutation_p_value,
)

STAGE27_SCHEMA_VERSION = "v2-stage27-prospective-signal-record-v1"
STAGE27_METADATA_SCHEMA_VERSION = "v2-stage27-metadata-v1"
STAGE27_SUMMARY_SCHEMA_VERSION = "v2-stage27-summary-v1"
STAGE27_EXPERIMENT = "stage27_prospective_signal_tracking"
STAGE27_ROOT = Path("data") / "prospective" / "stage27"
STAGE27_SIGNALS = (
    "production_pair_lr",
    "pair_strength_direct",
    "frequency_20",
    "paired_random",
)
STAGE27_PRIMARY_ENDPOINTS = (
    "mean_winner_rank_advantage",
    "top5_capture_advantage",
    "top15_capture_advantage",
)
STATUS_FROZEN = "FROZEN"
STATUS_EVALUATED = "EVALUATED"
STATUS_MISSED = "MISSED_PROSPECTIVE_FREEZE"


@dataclass(frozen=True, slots=True)
class SignalRanking:
    signal_id: str
    description: str
    definition_version: str
    ranking: tuple[int, ...]
    top5: tuple[int, ...]
    top10: tuple[int, ...]
    top15: tuple[int, ...]
    top20: tuple[int, ...]
    configuration: dict[str, Any]
    ranking_hash: str


@dataclass(frozen=True, slots=True)
class Stage27Record:
    schema_version: str
    experiment: str
    lottery: str
    draw_number: int
    draw_date: str
    status: str
    created_at: str
    history_cutoff_draw: int
    history_cutoff_date: str
    history_dataset_hash: str
    prospective_start_draw: int
    signals: dict[str, SignalRanking]
    evaluation: dict[str, Any] | None
    freeze_hash: str
    evaluation_hash: str | None


@dataclass(frozen=True, slots=True)
class FreezeResult:
    status: str
    record: Stage27Record | None
    record_path: str | None
    existing_record: bool
    target_result_absent: bool
    missed_draws: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluated_count: int
    pending_count: int
    skipped_count: int
    evaluated_draws: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Stage27CycleResult:
    lottery: str
    prospective_start_draw: int
    latest_history_draw: int
    evaluated: EvaluationResult
    freeze: FreezeResult
    summary_path: str
    warnings: tuple[str, ...]


def run_stage27_cycle(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
    seed: int = DEFAULT_STAGE05_SEED,
    now: datetime | None = None,
) -> Stage27CycleResult:
    _require_mini(lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    metadata = initialize_stage27(ordered, lottery, root=root, initialized_at=now)
    missed = record_missed_draws(ordered, lottery, root=root)
    evaluated = evaluate_stage27_records(ordered, lottery, root=root)
    freeze = freeze_next_stage27_record(
        ordered,
        lottery,
        root=root,
        seed=seed,
        created_at=now,
    )
    summary_path = save_stage27_summary(rebuild_stage27_summary(lottery, root=root), root=root)
    warnings = tuple(f"missed prospective draw #{draw}" for draw in missed)
    return Stage27CycleResult(
        lottery=str(lottery.code),
        prospective_start_draw=int(metadata["prospective_start_draw"]),
        latest_history_draw=ordered[-1].draw_number,
        evaluated=evaluated,
        freeze=freeze,
        summary_path=str(summary_path),
        warnings=warnings,
    )


def initialize_stage27(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
    initialized_at: datetime | None = None,
) -> dict[str, Any]:
    _require_mini(lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    if not ordered:
        raise ResearchValidationError("Stage 27 requires existing Mini Loto history")
    path = _metadata_path(root, lottery)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    latest = ordered[-1]
    timestamp = _timestamp(initialized_at)
    payload = {
        "schema_version": STAGE27_METADATA_SCHEMA_VERSION,
        "experiment": STAGE27_EXPERIMENT,
        "lottery": str(lottery.code),
        "initialized_at": timestamp,
        "latest_known_draw_at_initialization": latest.draw_number,
        "latest_known_date_at_initialization": latest.draw_date.isoformat(),
        "prospective_start_draw": latest.draw_number + 1,
        "missed_draws": (),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return payload


def freeze_next_stage27_record(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
    seed: int = DEFAULT_STAGE05_SEED,
    created_at: datetime | None = None,
) -> FreezeResult:
    ordered = validate_lottery_dataset(draws, lottery)
    target = ordered[-1].draw_number + 1
    return freeze_stage27_record(
        ordered,
        lottery,
        target_draw_number=target,
        root=root,
        seed=seed,
        created_at=created_at,
    )


def freeze_stage27_record(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    target_draw_number: int,
    root: str | Path = STAGE27_ROOT,
    seed: int = DEFAULT_STAGE05_SEED,
    created_at: datetime | None = None,
) -> FreezeResult:
    _require_mini(lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    latest = ordered[-1]
    metadata = initialize_stage27(ordered, lottery, root=root, initialized_at=created_at)
    missed = tuple(
        draw.draw_number
        for draw in ordered
        if draw.draw_number >= int(metadata["prospective_start_draw"])
        and not stage27_record_path(root, lottery, draw.draw_number).exists()
    )
    if target_draw_number <= latest.draw_number:
        return FreezeResult(STATUS_MISSED, None, None, False, False, missed)
    path = stage27_record_path(root, lottery, target_draw_number)
    if path.exists():
        existing = load_stage27_record(path)
        if existing.freeze_hash != stage27_freeze_hash(existing):
            raise ResearchValidationError(
                f"Stage 27 record #{target_draw_number} has invalid freeze hash"
            )
        return FreezeResult(STATUS_FROZEN, existing, str(path), True, True, missed)
    target_date = _target_date_from_latest(latest, lottery, target_draw_number)
    record = build_stage27_record(
        ordered,
        lottery,
        target_draw_number=target_draw_number,
        target_draw_date=target_date.isoformat(),
        prospective_start_draw=int(metadata["prospective_start_draw"]),
        seed=seed,
        created_at=created_at,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(record), encoding="utf-8")
    return FreezeResult(STATUS_FROZEN, record, str(path), False, True, missed)


def build_stage27_record(
    history: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    target_draw_number: int,
    target_draw_date: str,
    prospective_start_draw: int,
    seed: int,
    created_at: datetime | None = None,
) -> Stage27Record:
    _require_mini(lottery)
    latest = history[-1]
    signals = build_signal_rankings(
        history,
        lottery,
        target_draw_number=target_draw_number,
        target_draw_date=target_draw_date,
        seed=seed,
    )
    base = Stage27Record(
        schema_version=STAGE27_SCHEMA_VERSION,
        experiment=STAGE27_EXPERIMENT,
        lottery=str(lottery.code),
        draw_number=target_draw_number,
        draw_date=target_draw_date,
        status=STATUS_FROZEN,
        created_at=_timestamp(created_at),
        history_cutoff_draw=latest.draw_number,
        history_cutoff_date=latest.draw_date.isoformat(),
        history_dataset_hash=calculate_dataset_hash(history),
        prospective_start_draw=prospective_start_draw,
        signals=signals,
        evaluation=None,
        freeze_hash="",
        evaluation_hash=None,
    )
    return replace(base, freeze_hash=stage27_freeze_hash(base))


def build_signal_rankings(
    history: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    *,
    target_draw_number: int,
    target_draw_date: str,
    seed: int,
) -> dict[str, SignalRanking]:
    strategy = production_strategy_for_lottery(lottery)
    production_scores = _score_future_numbers(
        history,
        lottery,
        strategy,
        seed=seed,
        target_draw_number=target_draw_number,
        target_draw_date=datetime.fromisoformat(target_draw_date).date(),
    )
    state = _NumberFeatureState(lottery)
    for draw in history:
        state.add_draw(draw)
    pair_scores = {
        number: state.feature_values_for_number(number)["pair_strength_rate"]
        for number in range(lottery.number_min, lottery.number_max + 1)
    }
    frequency_scores = {
        number: state.feature_values_for_number(number)["frequency_20"]
        for number in range(lottery.number_min, lottery.number_max + 1)
    }
    random_order = _random_order(lottery, seed, target_draw_number)
    return {
        "production_pair_lr": _signal(
            "production_pair_lr",
            "Current production Logistic Regression pair_only ranking model.",
            rank_numbers_from_scores(production_scores),
            {
                "model": strategy.model_name,
                "feature_group": strategy.feature_group,
                "feature_names": strategy.feature_names,
                "portfolio_method": strategy.portfolio_method,
                "seed": seed,
            },
        ),
        "pair_strength_direct": _signal(
            "pair_strength_direct",
            "Direct ranking by pair_strength_rate.",
            rank_numbers_from_scores(pair_scores),
            {
                "feature": "pair_strength_rate",
                "feature_group": "pair_only",
                "direction": "higher",
            },
        ),
        "frequency_20": _signal(
            "frequency_20",
            "Direct ranking by prior 20-draw main-number frequency.",
            rank_numbers_from_scores(frequency_scores),
            {
                "feature": "frequency_20",
                "direction": "higher",
                "lookback_draws": 20,
            },
        ),
        "paired_random": _signal(
            "paired_random",
            "Deterministic random ranking control frozen before result.",
            {number: index + 1 for index, number in enumerate(random_order)},
            {
                "control_seed": _random_control_seed(seed, lottery, target_draw_number),
                "global_seed": seed,
            },
        ),
    }


def evaluate_stage27_records(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
) -> EvaluationResult:
    _require_mini(lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    by_number = {draw.draw_number: draw for draw in ordered}
    evaluated: list[int] = []
    pending = 0
    skipped = 0
    for path in _record_paths(root, lottery):
        record = load_stage27_record(path)
        actual = by_number.get(record.draw_number)
        if actual is None:
            pending += 1
            continue
        updated = evaluated_stage27_record(record, actual, lottery)
        if record.status == STATUS_EVALUATED:
            skipped += 1
        else:
            path.write_text(research_result_json(updated), encoding="utf-8")
            evaluated.append(record.draw_number)
    return EvaluationResult(
        evaluated_count=len(evaluated),
        pending_count=pending,
        skipped_count=skipped,
        evaluated_draws=tuple(evaluated),
    )


def evaluated_stage27_record(
    record: Stage27Record,
    actual: HistoricalDraw,
    lottery: LotteryDefinition,
) -> Stage27Record:
    if record.draw_number != actual.draw_number:
        raise ResearchValidationError("Stage 27 actual draw does not match record")
    if record.freeze_hash != stage27_freeze_hash(record):
        raise ResearchValidationError("Stage 27 freeze hash conflict")
    evaluated_at = (
        str(record.evaluation["evaluated_at"])
        if record.evaluation is not None and "evaluated_at" in record.evaluation
        else datetime.now(UTC).isoformat()
    )
    evaluation = _evaluation_payload(record, actual, lottery, evaluated_at=evaluated_at)
    evaluation_hash = _stable_hash(evaluation)
    if record.status == STATUS_EVALUATED:
        if record.evaluation_hash != evaluation_hash:
            raise ResearchValidationError("Stage 27 evaluation hash conflict")
        return record
    return replace(
        record,
        status=STATUS_EVALUATED,
        evaluation=evaluation,
        evaluation_hash=evaluation_hash,
    )


def rebuild_stage27_summary(
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
    seed: int = DEFAULT_STAGE05_SEED,
    bootstrap_replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
) -> dict[str, Any]:
    _require_mini(lottery)
    records = tuple(load_stage27_record(path) for path in _record_paths(root, lottery))
    evaluated = tuple(record for record in records if record.status == STATUS_EVALUATED)
    metadata = _load_metadata_if_exists(root, lottery)
    signals = {
        signal_id: _signal_summary(
            signal_id,
            records,
            seed=seed,
            bootstrap_replications=bootstrap_replications,
        )
        for signal_id in STAGE27_SIGNALS
    }
    return {
        "schema_version": STAGE27_SUMMARY_SCHEMA_VERSION,
        "experiment": STAGE27_EXPERIMENT,
        "lottery": str(lottery.code),
        "prospective_start_draw": None if metadata is None else metadata["prospective_start_draw"],
        "frozen_draw_count": len(records),
        "evaluated_draw_count": len(evaluated),
        "pending_draw_count": sum(record.status == STATUS_FROZEN for record in records),
        "missed_draws": () if metadata is None else tuple(metadata.get("missed_draws", ())),
        "signals": signals,
        "champion_direct_equality": _champion_direct_summary(evaluated),
        "classification": _evidence_classification(len(evaluated), 0.0, 1.0),
        "source": "derived from immutable Stage 27 draw records",
    }


def save_stage27_summary(
    payload: dict[str, Any],
    *,
    root: str | Path = STAGE27_ROOT,
) -> Path:
    path = _lottery_dir(root, MINI_LOTO) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(payload), encoding="utf-8")
    return path


def record_missed_draws(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    root: str | Path = STAGE27_ROOT,
) -> tuple[int, ...]:
    metadata = initialize_stage27(draws, lottery, root=root)
    existing = {int(path.stem) for path in _record_paths(root, lottery)}
    ordered = validate_lottery_dataset(draws, lottery)
    missed = tuple(
        draw.draw_number
        for draw in ordered
        if draw.draw_number >= int(metadata["prospective_start_draw"])
        and draw.draw_number not in existing
    )
    merged = tuple(sorted({*metadata.get("missed_draws", ()), *missed}))
    if tuple(metadata.get("missed_draws", ())) != merged:
        payload = {**metadata, "missed_draws": merged}
        _metadata_path(root, lottery).write_text(research_result_json(payload), encoding="utf-8")
    return missed


def load_stage27_record(path: str | Path) -> Stage27Record:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _record_from_payload(payload)


def stage27_record_path(root: str | Path, lottery: LotteryDefinition, draw_number: int) -> Path:
    return _lottery_dir(root, lottery) / f"{draw_number}.json"


def stage27_freeze_hash(record: Stage27Record) -> str:
    return _stable_hash(_freeze_payload(record))


def _evaluation_payload(
    record: Stage27Record,
    actual: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    evaluated_at: str,
) -> dict[str, Any]:
    signal_results = {
        signal_id: _evaluate_signal(signal, actual) for signal_id, signal in record.signals.items()
    }
    random_result = signal_results["paired_random"]
    comparisons = {
        signal_id: _paired_comparison(signal_result, random_result)
        for signal_id, signal_result in signal_results.items()
        if signal_id != "paired_random"
    }
    champion = record.signals["production_pair_lr"]
    direct = record.signals["pair_strength_direct"]
    equality = {
        "full_rank_equality": champion.ranking == direct.ranking,
        "spearman": _spearman(
            _ranks_from_order(champion.ranking), _ranks_from_order(direct.ranking)
        ),
        "top5_equality": champion.top5 == direct.top5,
        "top15_equality": champion.top15 == direct.top15,
    }
    return {
        "evaluated_at": evaluated_at,
        "actual_draw_number": actual.draw_number,
        "actual_draw_date": actual.draw_date.isoformat(),
        "actual_main_numbers": actual.main_numbers,
        "actual_bonus_numbers": actual.bonus_numbers,
        "signal_results": signal_results,
        "paired_comparisons_vs_random": comparisons,
        "production_vs_pair_strength_direct": equality,
        "bonus_excluded_from_main_capture": True,
        "lottery": str(lottery.code),
    }


def _evaluate_signal(signal: SignalRanking, actual: HistoricalDraw) -> dict[str, Any]:
    ranks = _ranks_from_order(signal.ranking)
    winner_ranks = tuple(ranks[number] for number in actual.main_numbers)
    return {
        "signal_id": signal.signal_id,
        "winner_ranks": {number: ranks[number] for number in actual.main_numbers},
        "mean_winner_rank": mean(winner_ranks),
        "median_winner_rank": median(winner_ranks),
        "best_winner_rank": min(winner_ranks),
        "worst_winner_rank": max(winner_ranks),
        "top5_captured_count": sum(rank <= 5 for rank in winner_ranks),
        "top10_captured_count": sum(rank <= 10 for rank in winner_ranks),
        "top15_captured_count": sum(rank <= 15 for rank in winner_ranks),
        "top20_captured_count": sum(rank <= 20 for rank in winner_ranks),
        "top5_capture": sum(rank <= 5 for rank in winner_ranks) / 5,
        "top10_capture": sum(rank <= 10 for rank in winner_ranks) / 5,
        "top15_capture": sum(rank <= 15 for rank in winner_ranks) / 5,
        "top20_capture": sum(rank <= 20 for rank in winner_ranks) / 5,
    }


def _paired_comparison(signal: dict[str, Any], random_signal: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_winner_rank_advantage": random_signal["mean_winner_rank"]
        - signal["mean_winner_rank"],
        "top5_capture_advantage": signal["top5_captured_count"]
        - random_signal["top5_captured_count"],
        "top15_capture_advantage": signal["top15_captured_count"]
        - random_signal["top15_captured_count"],
    }


def _signal_summary(
    signal_id: str,
    records: tuple[Stage27Record, ...],
    *,
    seed: int,
    bootstrap_replications: int,
) -> dict[str, Any]:
    evaluated_records = tuple(record for record in records if record.status == STATUS_EVALUATED)
    values = [
        record.evaluation["signal_results"][signal_id]  # type: ignore[index]
        for record in evaluated_records
        if record.evaluation is not None
    ]
    if signal_id == "paired_random":
        comparisons: list[dict[str, float]] = []
    else:
        comparisons = [
            record.evaluation["paired_comparisons_vs_random"][signal_id]  # type: ignore[index]
            for record in evaluated_records
            if record.evaluation is not None
        ]
    mean_diffs = tuple(item["mean_winner_rank_advantage"] for item in comparisons)
    top5_diffs = tuple(item["top5_capture_advantage"] for item in comparisons)
    top15_diffs = tuple(item["top15_capture_advantage"] for item in comparisons)
    raw_p = (
        paired_permutation_p_value(
            mean_diffs,
            seed=_derived_seed(seed, f"stage27-{signal_id}-mean-rank"),
            replications=bootstrap_replications,
        )
        if len(mean_diffs) >= 10
        else None
    )
    ci = (
        bootstrap_confidence_interval(
            mean_diffs,
            seed=_derived_seed(seed, f"stage27-{signal_id}-mean-rank-ci"),
            replications=bootstrap_replications,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        )
        if len(mean_diffs) >= 10
        else None
    )
    effect = _effect_size(mean_diffs)
    return {
        "frozen_draw_count": sum(signal_id in record.signals for record in records),
        "evaluated_draw_count": len(values),
        "pending_draw_count": sum(
            record.status == STATUS_FROZEN and signal_id in record.signals for record in records
        ),
        "mean_winner_rank": _avg(values, "mean_winner_rank"),
        "random_mean_winner_rank": None
        if signal_id == "paired_random"
        else _avg(
            [
                record.evaluation["signal_results"]["paired_random"]  # type: ignore[index]
                for record in evaluated_records
                if record.evaluation is not None
            ],
            "mean_winner_rank",
        ),
        "mean_rank_advantage": mean(mean_diffs) if mean_diffs else None,
        "top5_capture_average": _avg(values, "top5_capture"),
        "random_top5_average": None
        if signal_id == "paired_random"
        else _avg(
            [
                record.evaluation["signal_results"]["paired_random"]  # type: ignore[index]
                for record in evaluated_records
                if record.evaluation is not None
            ],
            "top5_capture",
        ),
        "top15_capture_average": _avg(values, "top15_capture"),
        "random_top15_average": None
        if signal_id == "paired_random"
        else _avg(
            [
                record.evaluation["signal_results"]["paired_random"]  # type: ignore[index]
                for record in evaluated_records
                if record.evaluation is not None
            ],
            "top15_capture",
        ),
        "wins_ties_losses": {
            "mean_winner_rank": _wins_ties_losses(mean_diffs),
            "top5_capture": _wins_ties_losses(top5_diffs),
            "top15_capture": _wins_ties_losses(top15_diffs),
        },
        "cumulative_paired_mean_difference": mean(mean_diffs) if mean_diffs else None,
        "bootstrap_95_ci": ci,
        "paired_permutation_p_value": raw_p,
        "effect_size": effect,
        "classification": _evidence_classification(
            len(values),
            mean(mean_diffs) if mean_diffs else 0,
            raw_p,
        ),
    }


def _champion_direct_summary(evaluated_records: tuple[Stage27Record, ...]) -> dict[str, Any]:
    equality = [
        record.evaluation["production_vs_pair_strength_direct"]  # type: ignore[index]
        for record in evaluated_records
        if record.evaluation is not None
    ]
    identical = sum(item["full_rank_equality"] for item in equality)
    first_divergence = next(
        (
            record.draw_number
            for record in evaluated_records
            if record.evaluation is not None
            and not record.evaluation["production_vs_pair_strength_direct"]["full_rank_equality"]
        ),
        None,
    )
    return {
        "identical_ranking_draws": identical,
        "evaluated_draws": len(equality),
        "fraction_identical": identical / len(equality) if equality else None,
        "first_divergence_draw": first_divergence,
    }


def _signal(
    signal_id: str,
    description: str,
    ranks: dict[int, int],
    configuration: dict[str, Any],
) -> SignalRanking:
    ranking = tuple(number for number, _rank in sorted(ranks.items(), key=lambda item: item[1]))
    payload = {
        "signal_id": signal_id,
        "description": description,
        "definition_version": "stage27-v1",
        "ranking": ranking,
        "configuration": configuration,
    }
    return SignalRanking(
        signal_id=signal_id,
        description=description,
        definition_version="stage27-v1",
        ranking=ranking,
        top5=ranking[:5],
        top10=ranking[:10],
        top15=ranking[:15],
        top20=ranking[:20],
        configuration=configuration,
        ranking_hash=_stable_hash(payload),
    )


def _random_order(
    lottery: LotteryDefinition,
    seed: int,
    target_draw_number: int,
) -> tuple[int, ...]:
    rng = random.Random(_random_control_seed(seed, lottery, target_draw_number))
    return tuple(
        rng.sample(
            range(lottery.number_min, lottery.number_max + 1),
            lottery.number_max - lottery.number_min + 1,
        )
    )


def _random_control_seed(seed: int, lottery: LotteryDefinition, target_draw_number: int) -> int:
    return _derived_seed(seed, f"{lottery.code}-{target_draw_number}-paired_random")


def _target_date_from_latest(
    latest: HistoricalDraw,
    lottery: LotteryDefinition,
    target_draw_number: int,
) -> date:
    draw_date = latest.draw_date
    draw_number = latest.draw_number
    while draw_number < target_draw_number:
        draw_date = next_scheduled_draw_date(draw_date, lottery)
        draw_number += 1
    return draw_date


def _freeze_payload(record: Stage27Record) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "experiment": record.experiment,
        "lottery": record.lottery,
        "draw_number": record.draw_number,
        "draw_date": record.draw_date,
        "created_at": record.created_at,
        "history_cutoff_draw": record.history_cutoff_draw,
        "history_cutoff_date": record.history_cutoff_date,
        "history_dataset_hash": record.history_dataset_hash,
        "prospective_start_draw": record.prospective_start_draw,
        "signals": record.signals,
    }


def _record_from_payload(payload: dict[str, Any]) -> Stage27Record:
    signals = {
        signal_id: SignalRanking(
            signal_id=signal["signal_id"],
            description=signal["description"],
            definition_version=signal["definition_version"],
            ranking=tuple(signal["ranking"]),
            top5=tuple(signal["top5"]),
            top10=tuple(signal["top10"]),
            top15=tuple(signal["top15"]),
            top20=tuple(signal["top20"]),
            configuration=signal["configuration"],
            ranking_hash=signal["ranking_hash"],
        )
        for signal_id, signal in payload["signals"].items()
    }
    return Stage27Record(
        schema_version=payload["schema_version"],
        experiment=payload["experiment"],
        lottery=payload["lottery"],
        draw_number=int(payload["draw_number"]),
        draw_date=payload["draw_date"],
        status=payload["status"],
        created_at=payload["created_at"],
        history_cutoff_draw=int(payload["history_cutoff_draw"]),
        history_cutoff_date=payload["history_cutoff_date"],
        history_dataset_hash=payload["history_dataset_hash"],
        prospective_start_draw=int(payload["prospective_start_draw"]),
        signals=signals,
        evaluation=payload["evaluation"],
        freeze_hash=payload["freeze_hash"],
        evaluation_hash=payload["evaluation_hash"],
    )


def _record_paths(root: str | Path, lottery: LotteryDefinition) -> tuple[Path, ...]:
    directory = _lottery_dir(root, lottery)
    if not directory.exists():
        return ()
    return tuple(
        path
        for path in sorted(directory.glob("*.json"))
        if path.name not in {"metadata.json", "summary.json"}
    )


def _metadata_path(root: str | Path, lottery: LotteryDefinition) -> Path:
    return _lottery_dir(root, lottery) / "metadata.json"


def _load_metadata_if_exists(root: str | Path, lottery: LotteryDefinition) -> dict[str, Any] | None:
    path = _metadata_path(root, lottery)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _lottery_dir(root: str | Path, lottery: LotteryDefinition) -> Path:
    return Path(root) / str(lottery.code)


def _ranks_from_order(ranking: tuple[int, ...]) -> dict[int, int]:
    return {number: index + 1 for index, number in enumerate(ranking)}


def _spearman(left: dict[int, int], right: dict[int, int]) -> float:
    numbers = tuple(sorted(left))
    n = len(numbers)
    if n < 2:
        return 0.0
    squared = sum((left[number] - right[number]) ** 2 for number in numbers)
    return 1 - (6 * squared) / (n * (n * n - 1))


def _avg(values: list[dict[str, Any]], key: str) -> float | None:
    return mean(tuple(float(value[key]) for value in values)) if values else None


def _wins_ties_losses(values: tuple[float, ...]) -> dict[str, int]:
    return {
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def _effect_size(differences: tuple[float, ...]) -> EffectSize | None:
    if not differences:
        return None
    average = mean(differences)
    std = pstdev(differences) if len(differences) > 1 else 0.0
    return EffectSize(
        absolute_difference=average,
        relative_difference=None,
        standardized_mean_difference=0.0 if std == 0 else average / std,
    )


def _evidence_classification(
    evaluated_draws: int, mean_difference: float, p_value: float | None
) -> str:
    if evaluated_draws < 10:
        return "INSUFFICIENT_DATA"
    if evaluated_draws < 26:
        return "EARLY_TRACKING"
    if evaluated_draws < 50:
        if mean_difference > 0:
            return "PRELIMINARY_POSITIVE"
        if mean_difference < 0:
            return "PRELIMINARY_NEGATIVE"
        return "PRELIMINARY_NEUTRAL"
    if p_value is not None and p_value < 0.05 and mean_difference > 0:
        return "ELIGIBLE_FOR_REVIEW"
    return "PRELIMINARY_NEUTRAL"


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(research_result_json(payload).encode("utf-8")).hexdigest()


def _timestamp(value: datetime | None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def _require_mini(lottery: LotteryDefinition) -> None:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError(
            "Stage 27 prospective signal tracking supports MINI_LOTO only"
        )


def stage27_payload(result: Stage27CycleResult) -> dict[str, Any]:
    return to_jsonable(result)
