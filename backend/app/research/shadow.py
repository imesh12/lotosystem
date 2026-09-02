from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import LOTO6, MINI_LOTO
from backend.app.research.baseline_benchmark import (
    DEFAULT_STAGE05_SEED,
    generate_distinct_random_tickets,
)
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.history_import import canonical_history_path
from backend.app.research.ml_baseline import DEFAULT_ML_MIN_TRAINING_DRAWS
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.prize import match_ticket
from backend.app.research.production import (
    ProductionStrategyConfig,
    _generate_ranked_tickets,
    _model_parameters,
    _score_future_numbers,
    next_scheduled_draw_date,
)
from backend.app.research.prospective import (
    DEFAULT_PROSPECTIVE_RANDOM_REPLICATIONS,
    DIAGNOSIS_INSUFFICIENT_DATA,
    PROSPECTIVE_TARGET_DRAW_TIME,
    load_prospective_record,
    prospective_record_path,
)

SHADOW_ROOT = Path("data") / "shadow"
SHADOW_REGISTRY_SCHEMA_VERSION = "v2-stage21-shadow-registry-v1"
SHADOW_RECORD_SCHEMA_VERSION = "v2-stage21-shadow-prediction-v1"
SHADOW_SUMMARY_SCHEMA_VERSION = "v2-stage21-shadow-summary-v1"
JST = timezone(timedelta(hours=9), "Asia/Tokyo")

STATUS_REGISTERED = "REGISTERED"
STATUS_ACTIVE_SHADOW = "ACTIVE_SHADOW"
STATUS_PAUSED = "PAUSED"
STATUS_RETIRED = "RETIRED"
STATUS_ELIGIBLE_FOR_REVIEW = "ELIGIBLE_FOR_REVIEW"
CHALLENGER_STATUSES = (
    STATUS_REGISTERED,
    STATUS_ACTIVE_SHADOW,
    STATUS_PAUSED,
    STATUS_RETIRED,
    STATUS_ELIGIBLE_FOR_REVIEW,
)
SHADOW_PENDING = "PENDING"
SHADOW_EVALUATED = "EVALUATED"
TEST_MODEL_NAME = "test_deterministic"


@dataclass(frozen=True, slots=True)
class ShadowChallenger:
    challenger_id: str
    lottery: str
    status: str
    model: str
    feature_group: str
    feature_names: tuple[str, ...]
    feature_version: str
    portfolio_method: str
    portfolio_version: str
    config_version: str
    seed: int
    registered_at: str
    frozen_at: str | None
    prospective_start_draw: int | None
    minimum_evaluation_draws: int
    notes: str
    research_experiment_id: str | None
    research_experiment_status: str | None


@dataclass(frozen=True, slots=True)
class ShadowTicket:
    ticket_index: int
    numbers: tuple[int, ...]
    score: float


@dataclass(frozen=True, slots=True)
class ShadowPredictionRecord:
    schema_version: str
    challenger_id: str
    status: str
    lottery: str
    target_draw_number: int
    target_draw_date: str
    generated_at: str
    dataset_hash: str
    latest_source_draw_number: int
    latest_source_draw_date: str
    model: str
    model_parameters: dict[str, Any]
    feature_group: str
    feature_version: str
    feature_names: tuple[str, ...]
    portfolio_method: str
    portfolio_version: str
    seed: int
    ticket_count: int
    tickets: tuple[ShadowTicket, ...]
    ranking: dict[str, Any]
    prospective_eligible: bool
    result_linkage: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    champion_comparison: dict[str, Any] | None
    random_control: dict[str, Any] | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShadowGenerationResult:
    record: ShadowPredictionRecord
    record_path: str
    existing_record: bool


def load_shadow_registry(root: str | Path = SHADOW_ROOT) -> dict[str, Any]:
    path = _registry_path(root)
    if not path.exists():
        return {"schema_version": SHADOW_REGISTRY_SCHEMA_VERSION, "challengers": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_shadow_registry(registry: dict[str, Any], root: str | Path = SHADOW_ROOT) -> Path:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(research_result_json(registry), encoding="utf-8")
    return path


def register_shadow_challenger(
    *,
    challenger_id: str,
    lottery: LotteryDefinition,
    model: str,
    feature_group: str,
    portfolio_method: str = "top_ranked",
    status: str = STATUS_REGISTERED,
    seed: int = DEFAULT_STAGE05_SEED,
    minimum_evaluation_draws: int = 10,
    notes: str = "",
    research_experiment_id: str | None = None,
    research_experiment_status: str | None = None,
    root: str | Path = SHADOW_ROOT,
    override_retired_experiment: bool = False,
    registered_at: datetime | None = None,
) -> ShadowChallenger:
    _validate_challenger_id(challenger_id)
    if status not in CHALLENGER_STATUSES:
        raise ResearchValidationError(f"invalid shadow challenger status: {status}")
    if minimum_evaluation_draws <= 0:
        raise ResearchValidationError("minimum_evaluation_draws must be positive")
    if status == STATUS_ACTIVE_SHADOW and research_experiment_status == "RETIRE":
        if not override_retired_experiment:
            raise ResearchValidationError("retired experiment cannot become ACTIVE_SHADOW")
    if feature_group not in FEATURE_GROUPS:
        raise ResearchValidationError(f"unknown feature group: {feature_group}")
    if model not in {"logistic_regression", "random_forest", TEST_MODEL_NAME}:
        raise ResearchValidationError(f"unsupported shadow challenger model: {model}")
    registry = load_shadow_registry(root)
    if any(item["challenger_id"] == challenger_id for item in registry.get("challengers", ())):
        raise ResearchValidationError(f"duplicate shadow challenger: {challenger_id}")
    now = (registered_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    challenger = ShadowChallenger(
        challenger_id=challenger_id,
        lottery=str(lottery.code),
        status=status,
        model=model,
        feature_group=feature_group,
        feature_names=FEATURE_GROUPS[feature_group],
        feature_version="number-features-v2",
        portfolio_method=portfolio_method,
        portfolio_version="shadow-portfolio-v1",
        config_version="stage21-shadow-config-v1",
        seed=seed,
        registered_at=now,
        frozen_at=now if status == STATUS_ACTIVE_SHADOW else None,
        prospective_start_draw=None,
        minimum_evaluation_draws=minimum_evaluation_draws,
        notes=notes,
        research_experiment_id=research_experiment_id,
        research_experiment_status=research_experiment_status,
    )
    registry["challengers"].append(to_jsonable(challenger))
    save_shadow_registry(registry, root)
    return challenger


def shadow_registry_payload(
    *,
    lottery: LotteryDefinition | None = None,
    root: str | Path = SHADOW_ROOT,
) -> dict[str, Any]:
    registry = load_shadow_registry(root)
    challengers = tuple(
        item
        for item in registry.get("challengers", ())
        if lottery is None or item["lottery"] == str(lottery.code)
    )
    return {
        "schema_version": registry["schema_version"],
        "challengers": challengers,
        "count": len(challengers),
    }


def generate_shadow_prediction(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    challenger_id: str,
    *,
    tickets_per_draw: int = 3,
    root: str | Path = SHADOW_ROOT,
    generated_at: datetime | None = None,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
) -> ShadowGenerationResult:
    if tickets_per_draw <= 0:
        raise ResearchValidationError("tickets_per_draw must be positive")
    challenger = _load_challenger(root, challenger_id, lottery)
    if challenger.status != STATUS_ACTIVE_SHADOW:
        raise ResearchValidationError("shadow challenger is not ACTIVE_SHADOW")
    ordered = validate_lottery_dataset(draws, lottery)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough history to generate shadow prediction")
    latest = ordered[-1]
    target_draw_number = latest.draw_number + 1
    target_draw_date = next_scheduled_draw_date(latest.draw_date, lottery)
    record_path = shadow_record_path(root, lottery, challenger_id, target_draw_number)
    if record_path.exists():
        return ShadowGenerationResult(
            record=load_shadow_record(record_path),
            record_path=str(record_path),
            existing_record=True,
        )
    seed = challenger.seed
    scores = _shadow_scores(ordered, lottery, challenger, target_draw_number, target_draw_date)
    tickets = _shadow_tickets(scores, lottery, tickets_per_draw, challenger.portfolio_method)
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    record = ShadowPredictionRecord(
        schema_version=SHADOW_RECORD_SCHEMA_VERSION,
        challenger_id=challenger_id,
        status=SHADOW_PENDING,
        lottery=str(lottery.code),
        target_draw_number=target_draw_number,
        target_draw_date=target_draw_date.isoformat(),
        generated_at=timestamp,
        dataset_hash=calculate_dataset_hash(ordered),
        latest_source_draw_number=latest.draw_number,
        latest_source_draw_date=latest.draw_date.isoformat(),
        model=challenger.model,
        model_parameters=_shadow_model_parameters(challenger.model, seed),
        feature_group=challenger.feature_group,
        feature_version=challenger.feature_version,
        feature_names=challenger.feature_names,
        portfolio_method=challenger.portfolio_method,
        portfolio_version=challenger.portfolio_version,
        seed=seed,
        ticket_count=tickets_per_draw,
        tickets=tickets,
        ranking=_ranking_metadata(scores),
        prospective_eligible=_generated_before_target(timestamp, target_draw_date),
        result_linkage=None,
        evaluation=None,
        champion_comparison=None,
        random_control=None,
        warnings=(
            "Shadow prediction is research-only and is not a production ticket.",
            "Shadow records do not affect production predictions, settlements, or email.",
        ),
    )
    if not record.prospective_eligible:
        raise ResearchValidationError("shadow prediction generated after target draw")
    save_shadow_record(record, record_path)
    _update_shadow_ledger(root, lottery, challenger_id)
    return ShadowGenerationResult(record, str(record_path), existing_record=False)


def evaluate_shadow_predictions(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    challenger_id: str | None = None,
    root: str | Path = SHADOW_ROOT,
    prospective_root: str | Path = Path("data/prospective"),
    random_replications: int = DEFAULT_PROSPECTIVE_RANDOM_REPLICATIONS,
) -> dict[str, Any]:
    ordered = validate_lottery_dataset(draws, lottery)
    by_number = {draw.draw_number: draw for draw in ordered}
    evaluated: list[dict[str, Any]] = []
    pending = 0
    skipped = 0
    for path in _shadow_record_paths(root, lottery, challenger_id):
        record = load_shadow_record(path)
        if record.status == SHADOW_EVALUATED:
            skipped += 1
            continue
        target_draw = by_number.get(record.target_draw_number)
        if target_draw is None:
            pending += 1
            continue
        if not record.prospective_eligible or not _generated_before_target(
            record.generated_at,
            target_draw.draw_date,
        ):
            raise ResearchValidationError("shadow record is not prospectively eligible")
        evaluated_record = _evaluated_shadow_record(
            record,
            target_draw,
            lottery,
            prospective_root=prospective_root,
            random_replications=random_replications,
        )
        save_shadow_record(evaluated_record, path)
        _update_shadow_ledger(root, lottery, record.challenger_id)
        evaluated.append(_shadow_record_summary(evaluated_record))
    return {
        "status": "ok",
        "lottery": str(lottery.code),
        "evaluated_count": len(evaluated),
        "pending_count": pending,
        "skipped_count": skipped,
        "evaluated": tuple(evaluated),
    }


def shadow_summary(
    *,
    lottery: LotteryDefinition | None = None,
    challenger_id: str | None = None,
    root: str | Path = SHADOW_ROOT,
    save: bool = False,
) -> dict[str, Any]:
    lotteries = (LOTO6, MINI_LOTO) if lottery is None else (lottery,)
    challenger_summaries: list[dict[str, Any]] = []
    for selected in lotteries:
        for challenger in _challengers(root, selected):
            if challenger_id is not None and challenger.challenger_id != challenger_id:
                continue
            records = tuple(
                load_shadow_record(path)
                for path in _shadow_record_paths(root, selected, challenger.challenger_id)
            )
            challenger_summaries.append(_summary_for_challenger(challenger, records))
    payload = {
        "schema_version": SHADOW_SUMMARY_SCHEMA_VERSION,
        "challengers": tuple(challenger_summaries),
        "production_safety": {
            "auto_promotion": False,
            "writes_production_predictions": False,
            "writes_production_settlements": False,
            "sends_email": False,
        },
    }
    if save:
        path = Path(root) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(research_result_json(payload), encoding="utf-8")
        payload["summary_path"] = str(path)
    return payload


def save_shadow_record(record: ShadowPredictionRecord, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_shadow_record(destination)
        if existing.status == SHADOW_PENDING and record.status == SHADOW_EVALUATED:
            destination.write_text(research_result_json(record), encoding="utf-8")
            return destination
        if _shadow_scientific_payload(existing) != _shadow_scientific_payload(record):
            raise ResearchValidationError(
                f"conflicting shadow record for {record.challenger_id} "
                f"{record.lottery} #{record.target_draw_number}"
            )
        return destination
    destination.write_text(research_result_json(record), encoding="utf-8")
    return destination


def load_shadow_record(path: str | Path) -> ShadowPredictionRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ShadowPredictionRecord(
        schema_version=payload["schema_version"],
        challenger_id=payload["challenger_id"],
        status=payload["status"],
        lottery=payload["lottery"],
        target_draw_number=int(payload["target_draw_number"]),
        target_draw_date=payload["target_draw_date"],
        generated_at=payload["generated_at"],
        dataset_hash=payload["dataset_hash"],
        latest_source_draw_number=int(payload["latest_source_draw_number"]),
        latest_source_draw_date=payload["latest_source_draw_date"],
        model=payload["model"],
        model_parameters=payload["model_parameters"],
        feature_group=payload["feature_group"],
        feature_version=payload["feature_version"],
        feature_names=tuple(payload["feature_names"]),
        portfolio_method=payload["portfolio_method"],
        portfolio_version=payload["portfolio_version"],
        seed=int(payload["seed"]),
        ticket_count=int(payload["ticket_count"]),
        tickets=tuple(
            ShadowTicket(
                ticket_index=int(ticket["ticket_index"]),
                numbers=tuple(ticket["numbers"]),
                score=float(ticket["score"]),
            )
            for ticket in payload["tickets"]
        ),
        ranking=payload["ranking"],
        prospective_eligible=bool(payload["prospective_eligible"]),
        result_linkage=payload["result_linkage"],
        evaluation=payload["evaluation"],
        champion_comparison=payload["champion_comparison"],
        random_control=payload["random_control"],
        warnings=tuple(payload["warnings"]),
    )


def shadow_record_path(
    root: str | Path,
    lottery: LotteryDefinition,
    challenger_id: str,
    draw_number: int,
) -> Path:
    return Path(root) / str(lottery.code) / challenger_id / f"{draw_number}.json"


def _registry_path(root: str | Path) -> Path:
    return Path(root) / "registry.json"


def _load_challenger(
    root: str | Path,
    challenger_id: str,
    lottery: LotteryDefinition,
) -> ShadowChallenger:
    for challenger in _challengers(root, lottery):
        if challenger.challenger_id == challenger_id:
            return challenger
    raise ResearchValidationError(f"unknown shadow challenger: {challenger_id}")


def _challengers(root: str | Path, lottery: LotteryDefinition) -> tuple[ShadowChallenger, ...]:
    return tuple(
        _challenger_from_payload(item)
        for item in load_shadow_registry(root).get("challengers", ())
        if item["lottery"] == str(lottery.code)
    )


def _challenger_from_payload(payload: dict[str, Any]) -> ShadowChallenger:
    return ShadowChallenger(
        challenger_id=payload["challenger_id"],
        lottery=payload["lottery"],
        status=payload["status"],
        model=payload["model"],
        feature_group=payload["feature_group"],
        feature_names=tuple(payload["feature_names"]),
        feature_version=payload["feature_version"],
        portfolio_method=payload["portfolio_method"],
        portfolio_version=payload["portfolio_version"],
        config_version=payload["config_version"],
        seed=int(payload["seed"]),
        registered_at=payload["registered_at"],
        frozen_at=payload["frozen_at"],
        prospective_start_draw=payload["prospective_start_draw"],
        minimum_evaluation_draws=int(payload["minimum_evaluation_draws"]),
        notes=payload["notes"],
        research_experiment_id=payload["research_experiment_id"],
        research_experiment_status=payload["research_experiment_status"],
    )


def _shadow_scores(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    challenger: ShadowChallenger,
    target_draw_number: int,
    target_draw_date,
) -> dict[int, float]:
    if challenger.model == TEST_MODEL_NAME:
        return {
            number: float(lottery.number_max + 1 - number)
            for number in range(lottery.number_min, lottery.number_max + 1)
        }
    strategy = ProductionStrategyConfig(
        model_name=challenger.model,
        feature_group=challenger.feature_group,
        feature_names=challenger.feature_names,
        feature_version=challenger.feature_version,
        portfolio_method=challenger.portfolio_method,
        portfolio_version=challenger.portfolio_version,
    )
    return _score_future_numbers(
        draws,
        lottery,
        strategy,
        seed=challenger.seed,
        target_draw_number=target_draw_number,
        target_draw_date=target_draw_date,
    )


def _shadow_tickets(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    tickets_per_draw: int,
    method: str,
) -> tuple[ShadowTicket, ...]:
    production_tickets = _generate_ranked_tickets(
        scores,
        lottery,
        tickets_per_draw=tickets_per_draw,
        method=method,
        candidate_pool_size=50,
    )
    return tuple(
        ShadowTicket(ticket.ticket_index, ticket.numbers, ticket.score)
        for ticket in production_tickets
    )


def _shadow_model_parameters(model: str, seed: int) -> dict[str, Any]:
    if model == TEST_MODEL_NAME:
        return {"type": TEST_MODEL_NAME, "random_state": seed}
    return _model_parameters(model, seed)


def _ranking_metadata(scores: dict[int, float]) -> dict[str, Any]:
    ranked = tuple(
        number for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )
    return {
        "top_numbers": ranked,
        "scores": {str(number): scores[number] for number in ranked},
        "rank_by_number": {str(number): index + 1 for index, number in enumerate(ranked)},
    }


def _generated_before_target(generated_at: str, target_date) -> bool:
    generated = datetime.fromisoformat(generated_at).astimezone(JST)
    if isinstance(target_date, str):
        target_day = datetime.fromisoformat(target_date).date()
    else:
        target_day = target_date
    target = datetime.combine(target_day, PROSPECTIVE_TARGET_DRAW_TIME, tzinfo=JST)
    return generated < target


def _evaluated_shadow_record(
    record: ShadowPredictionRecord,
    target_draw: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    prospective_root: str | Path,
    random_replications: int,
) -> ShadowPredictionRecord:
    results = []
    for ticket in record.tickets:
        match = match_ticket(ticket.numbers, target_draw, lottery)
        results.append(
            {
                "ticket_index": ticket.ticket_index,
                "numbers": ticket.numbers,
                "main_matches": match.main_match_count,
                "bonus_match": match.bonus_match_count,
                "prize_tier": match.prize_name,
                "qualifies_for_prize": match.qualifies_for_prize,
                "payout_yen": None,
            }
        )
    match_counts = tuple(int(result["main_matches"]) for result in results)
    evaluation = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "actual_draw_number": target_draw.draw_number,
        "actual_draw_date": target_draw.draw_date.isoformat(),
        "actual_main_numbers": target_draw.main_numbers,
        "actual_bonus_numbers": target_draw.bonus_numbers,
        "ticket_results": tuple(results),
        "best_ticket_matches": max(match_counts) if match_counts else 0,
        "total_portfolio_matches": sum(match_counts),
        "prize_qualified_tickets": sum(result["qualifies_for_prize"] for result in results),
    }
    evaluated_record = replace(record, evaluation=evaluation)
    champion = _champion_comparison(evaluated_record, prospective_root, lottery)
    random_control = _shadow_random_control(
        record,
        target_draw,
        lottery,
        production_total=evaluation["total_portfolio_matches"],
        production_best=evaluation["best_ticket_matches"],
        replications=random_replications,
    )
    return replace(
        record,
        status=SHADOW_EVALUATED,
        result_linkage={
            "canonical_history_draw": target_draw.draw_number,
            "source": "canonical_history",
        },
        evaluation=evaluation,
        champion_comparison=champion,
        random_control=random_control,
    )


def _champion_comparison(
    record: ShadowPredictionRecord,
    prospective_root: str | Path,
    lottery: LotteryDefinition,
) -> dict[str, Any] | None:
    path = prospective_record_path(prospective_root, lottery, record.target_draw_number)
    if not path.exists():
        return None
    champion = load_prospective_record(path)
    if record.evaluation is None:
        shadow_total = 0
        shadow_best = 0
    else:
        shadow_total = int(record.evaluation["total_portfolio_matches"])
        shadow_best = int(record.evaluation["best_ticket_matches"])
    return {
        "champion_prospective_path": str(path),
        "champion_best_ticket_matches": champion.portfolio["best_ticket_matches"],
        "champion_total_matches": champion.portfolio["total_main_matches_across_tickets"],
        "challenger_best_minus_champion": shadow_best - champion.portfolio["best_ticket_matches"],
        "challenger_total_minus_champion": shadow_total
        - champion.portfolio["total_main_matches_across_tickets"],
        "champion_top_capture": champion.ranking["top_capture"],
        "challenger_top_capture": _top_capture(record, champion.actual_result["main_numbers"]),
    }


def _top_capture(
    record: ShadowPredictionRecord, winners: tuple[int, ...] | list[int]
) -> dict[str, int]:
    top_counts = (5, 10, 15, 20) if record.lottery == "MINI_LOTO" else (6, 12, 18, 24)
    ranks = {int(number): int(rank) for number, rank in record.ranking["rank_by_number"].items()}
    return {
        f"top_{count}": sum(ranks[int(number)] <= count for number in winners)
        for count in top_counts
    }


def _shadow_random_control(
    record: ShadowPredictionRecord,
    draw: HistoricalDraw,
    lottery: LotteryDefinition,
    *,
    production_total: int,
    production_best: int,
    replications: int,
) -> dict[str, Any]:
    if replications <= 0:
        raise ResearchValidationError("random replications must be positive")
    best_values: list[int] = []
    total_values: list[int] = []
    prize_portfolios = 0
    for replication in range(replications):
        rng = random.Random(record.seed + record.target_draw_number * 10_000 + replication)
        tickets = generate_distinct_random_tickets(lottery, rng, record.ticket_count)
        matches = []
        prize = False
        for ticket in tickets:
            result = match_ticket(ticket, draw, lottery)
            matches.append(result.main_match_count)
            prize = prize or result.qualifies_for_prize
        best_values.append(max(matches))
        total_values.append(sum(matches))
        prize_portfolios += int(prize)
    return {
        "replications": replications,
        "seed": record.seed,
        "tickets_per_draw": record.ticket_count,
        "random_mean_best_ticket_matches": mean(best_values),
        "random_mean_total_portfolio_matches": mean(total_values),
        "random_prize_qualified_rate": prize_portfolios / replications,
        "challenger_best_percentile": _percentile_at_or_below(best_values, production_best),
        "challenger_total_percentile": _percentile_at_or_below(total_values, production_total),
        "challenger_minus_random_total": production_total - mean(total_values),
    }


def _summary_for_challenger(
    challenger: ShadowChallenger,
    records: tuple[ShadowPredictionRecord, ...],
) -> dict[str, Any]:
    evaluated = tuple(record for record in records if record.status == SHADOW_EVALUATED)
    pending = tuple(record for record in records if record.status == SHADOW_PENDING)
    totals = tuple(int(record.evaluation["total_portfolio_matches"]) for record in evaluated)
    best = tuple(int(record.evaluation["best_ticket_matches"]) for record in evaluated)
    random_totals = tuple(
        float(record.random_control["random_mean_total_portfolio_matches"])
        for record in evaluated
        if record.random_control
    )
    champion_diffs = tuple(
        float(record.champion_comparison["challenger_total_minus_champion"])
        for record in evaluated
        if record.champion_comparison
    )
    prize_flags = tuple(
        int(record.evaluation["prize_qualified_tickets"]) > 0 for record in evaluated
    )
    return {
        "challenger_id": challenger.challenger_id,
        "lottery": challenger.lottery,
        "status": challenger.status,
        "eligible_draws": len(evaluated),
        "evaluated_draws": len(evaluated),
        "pending_draws": len(pending),
        "average_best_ticket_matches": mean(best) if best else 0.0,
        "average_total_matches": mean(totals) if totals else 0.0,
        "prize_qualified_rate": sum(prize_flags) / len(prize_flags) if prize_flags else 0.0,
        "average_top_k_winner_capture": _average_top_capture(evaluated),
        "mean_difference_vs_champion": mean(champion_diffs) if champion_diffs else None,
        "mean_difference_vs_random": mean(
            tuple(left - right for left, right in zip(totals, random_totals, strict=True))
        )
        if random_totals
        else None,
        "prospective_sample_status": "READY_FOR_REVIEW"
        if len(evaluated) >= challenger.minimum_evaluation_draws
        else DIAGNOSIS_INSUFFICIENT_DATA,
        "conclusion": DIAGNOSIS_INSUFFICIENT_DATA
        if len(evaluated) < challenger.minimum_evaluation_draws
        else "ELIGIBLE_FOR_REVIEW",
        "auto_promotion": False,
    }


def _average_top_capture(records: tuple[ShadowPredictionRecord, ...]) -> float:
    captures = []
    for record in records:
        if record.evaluation is None:
            continue
        values = tuple(_top_capture(record, record.evaluation["actual_main_numbers"]).values())
        captures.append(mean(values) if values else 0.0)
    return mean(captures) if captures else 0.0


def _shadow_record_paths(
    root: str | Path,
    lottery: LotteryDefinition,
    challenger_id: str | None,
) -> tuple[Path, ...]:
    lottery_dir = Path(root) / str(lottery.code)
    if not lottery_dir.exists():
        return ()
    if challenger_id is not None:
        directory = lottery_dir / challenger_id
        return tuple(
            path for path in sorted(directory.glob("*.json")) if path.name != "ledger.json"
        )
    return tuple(
        path
        for directory in sorted(lottery_dir.iterdir())
        if directory.is_dir()
        for path in sorted(directory.glob("*.json"))
        if path.name != "ledger.json"
    )


def _update_shadow_ledger(root: str | Path, lottery: LotteryDefinition, challenger_id: str) -> Path:
    directory = Path(root) / str(lottery.code) / challenger_id
    records = tuple(
        load_shadow_record(path)
        for path in sorted(directory.glob("*.json"))
        if path.name != "ledger.json"
    )
    payload = {
        "schema_version": "v2-stage21-shadow-ledger-v1",
        "lottery": str(lottery.code),
        "challenger_id": challenger_id,
        "entries": tuple(_shadow_record_summary(record) for record in records),
    }
    path = directory / "ledger.json"
    path.write_text(research_result_json(payload), encoding="utf-8")
    return path


def _shadow_record_summary(record: ShadowPredictionRecord) -> dict[str, Any]:
    return {
        "challenger_id": record.challenger_id,
        "lottery": record.lottery,
        "target_draw_number": record.target_draw_number,
        "target_draw_date": record.target_draw_date,
        "status": record.status,
        "ticket_count": record.ticket_count,
        "prospective_eligible": record.prospective_eligible,
        "best_ticket_matches": None
        if record.evaluation is None
        else record.evaluation["best_ticket_matches"],
        "total_portfolio_matches": None
        if record.evaluation is None
        else record.evaluation["total_portfolio_matches"],
        "champion_comparison_ready": record.champion_comparison is not None,
    }


def _shadow_scientific_payload(record: ShadowPredictionRecord) -> dict[str, Any]:
    payload = to_jsonable(record)
    if payload.get("evaluation") is not None:
        payload["evaluation"] = dict(payload["evaluation"])
        payload["evaluation"].pop("evaluated_at", None)
    return payload


def _validate_challenger_id(challenger_id: str) -> None:
    if not challenger_id or any(not (char.isalnum() or char in "-_") for char in challenger_id):
        raise ResearchValidationError("challenger_id must use letters, numbers, '-' or '_'")


def _percentile_at_or_below(values: list[int], observed: int) -> float:
    return sum(value <= observed for value in values) / len(values) if values else 0.0


def load_default_history(lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
    return load_draws_csv(canonical_history_path(lottery), lottery)
