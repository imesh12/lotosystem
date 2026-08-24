from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import sklearn

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.baseline_benchmark import DEFAULT_STAGE05_SEED
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.feature_evaluation import FEATURE_GROUPS
from backend.app.research.ml_baseline import (
    DEFAULT_ML_MIN_TRAINING_DRAWS,
    DEFAULT_ML_REFIT_INTERVAL,
    FEATURE_VERSION_V2,
    NumberFeatureRow,
    _make_model,
    _NumberFeatureState,
    _scores_from_fitted_model,
    build_training_dataset,
    build_walk_forward_feature_blocks,
)
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.portfolio_evaluation import (
    DEFAULT_CANDIDATE_POOL_SIZE,
    PORTFOLIO_VERSION,
    construct_portfolio,
)
from backend.app.research.prize import match_ticket

STAGE10_SCHEMA_VERSION = "stage10-production-prediction-v1"
PREDICTION_STATUS_PENDING = "PENDING"
PREDICTION_STATUS_EVALUATED = "EVALUATED"
PREDICTION_ROOT = Path("data") / "predictions"


@dataclass(frozen=True, slots=True)
class ProductionStrategyConfig:
    model_name: str
    feature_group: str
    feature_names: tuple[str, ...]
    feature_version: str
    portfolio_method: str
    portfolio_version: str


@dataclass(frozen=True, slots=True)
class PredictionTicket:
    ticket_index: int
    numbers: tuple[int, ...]
    score: float


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    schema_version: str
    status: str
    lottery: str
    target_draw_number: int
    target_draw_date: str
    generated_at: str
    dataset_hash: str
    latest_source_draw_number: int
    latest_source_draw_date: str
    strategy: str
    model: str
    model_parameters: dict[str, Any]
    feature_version: str
    feature_group: str
    feature_names: tuple[str, ...]
    portfolio_method: str
    portfolio_version: str
    seed: int
    tickets_per_draw: int
    ticket_price_yen: int
    cost_yen: int
    gross_winnings_yen: None
    sklearn_version: str
    config: dict[str, Any]
    tickets: tuple[PredictionTicket, ...]
    evaluation: dict[str, Any] | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratePredictionResult:
    record: PredictionRecord
    record_path: str
    existing_record: bool
    lookahead_safe: bool


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluated_count: int
    pending_count: int
    skipped_count: int
    evaluated_paths: tuple[str, ...]


def production_strategy_for_lottery(lottery: LotteryDefinition) -> ProductionStrategyConfig:
    if str(lottery.code) == "LOTO6":
        return ProductionStrategyConfig(
            model_name="random_forest",
            feature_group="gap_only",
            feature_names=FEATURE_GROUPS["gap_only"],
            feature_version=FEATURE_VERSION_V2,
            portfolio_method="top_ranked",
            portfolio_version=PORTFOLIO_VERSION,
        )
    return ProductionStrategyConfig(
        model_name="logistic_regression",
        feature_group="pair_only",
        feature_names=FEATURE_GROUPS["pair_only"],
        feature_version=FEATURE_VERSION_V2,
        portfolio_method="top_ranked",
        portfolio_version=PORTFOLIO_VERSION,
    )


def generate_next_prediction(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    *,
    tickets_per_draw: int = 2,
    prediction_root: str | Path = PREDICTION_ROOT,
    generated_at: datetime | None = None,
    ml_min_training_draws: int = DEFAULT_ML_MIN_TRAINING_DRAWS,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> GeneratePredictionResult:
    if tickets_per_draw <= 0:
        raise ResearchValidationError("tickets_per_draw must be positive")
    ordered = validate_lottery_dataset(draws, lottery)
    if len(ordered) <= ml_min_training_draws:
        raise ResearchValidationError("not enough history to generate production prediction")
    latest = ordered[-1]
    target_draw_number = latest.draw_number + 1
    target_draw_date = next_scheduled_draw_date(latest.draw_date, lottery)
    record_path = prediction_record_path(prediction_root, lottery, target_draw_number)
    if record_path.exists():
        record = load_prediction_record(record_path)
        return GeneratePredictionResult(
            record=record,
            record_path=str(record_path),
            existing_record=True,
            lookahead_safe=_record_matches_history(record, ordered),
        )

    strategy = production_strategy_for_lottery(lottery)
    seed = config.seed if config.seed is not None else DEFAULT_STAGE05_SEED
    scores = _score_future_numbers(
        ordered,
        lottery,
        strategy,
        seed=seed,
        target_draw_number=target_draw_number,
        target_draw_date=target_draw_date,
    )
    tickets = _generate_ranked_tickets(
        scores,
        lottery,
        tickets_per_draw=tickets_per_draw,
        method=strategy.portfolio_method,
        candidate_pool_size=candidate_pool_size,
    )
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    record = PredictionRecord(
        schema_version=STAGE10_SCHEMA_VERSION,
        status=PREDICTION_STATUS_PENDING,
        lottery=str(lottery.code),
        target_draw_number=target_draw_number,
        target_draw_date=target_draw_date.isoformat(),
        generated_at=timestamp,
        dataset_hash=calculate_dataset_hash(ordered),
        latest_source_draw_number=latest.draw_number,
        latest_source_draw_date=latest.draw_date.isoformat(),
        strategy="conservative_stage09_selection",
        model=strategy.model_name,
        model_parameters=_model_parameters(strategy.model_name, seed),
        feature_version=strategy.feature_version,
        feature_group=strategy.feature_group,
        feature_names=strategy.feature_names,
        portfolio_method=strategy.portfolio_method,
        portfolio_version=strategy.portfolio_version,
        seed=seed,
        tickets_per_draw=tickets_per_draw,
        ticket_price_yen=lottery.ticket_price_yen,
        cost_yen=tickets_per_draw * lottery.ticket_price_yen,
        gross_winnings_yen=None,
        sklearn_version=sklearn.__version__,
        config={
            "ml_min_training_draws": ml_min_training_draws,
            "ml_refit_interval": DEFAULT_ML_REFIT_INTERVAL,
            "candidate_pool_size": candidate_pool_size,
        },
        tickets=tickets,
        evaluation=None,
        warnings=(
            "Paper-trading record only; no ticket purchase is performed.",
            "Generated tickets are research outputs, not guaranteed winning numbers.",
            "Payout amounts and ROI are not calculated.",
        ),
    )
    save_prediction_record(record, record_path)
    update_prediction_ledger(prediction_root, lottery)
    return GeneratePredictionResult(
        record=record,
        record_path=str(record_path),
        existing_record=False,
        lookahead_safe=True,
    )


def evaluate_pending_predictions(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    prediction_root: str | Path = PREDICTION_ROOT,
) -> EvaluationResult:
    ordered = validate_lottery_dataset(draws, lottery)
    draws_by_number = {draw.draw_number: draw for draw in ordered}
    evaluated_paths: list[str] = []
    pending_count = 0
    skipped_count = 0
    for path in sorted(prediction_lottery_dir(prediction_root, lottery).glob("*.json")):
        if path.name == "ledger.json":
            continue
        record = load_prediction_record(path)
        if record.status == PREDICTION_STATUS_EVALUATED:
            skipped_count += 1
            continue
        target_draw = draws_by_number.get(record.target_draw_number)
        if target_draw is None:
            pending_count += 1
            continue
        evaluated = _evaluated_record(record, target_draw, lottery)
        save_prediction_record(evaluated, path)
        evaluated_paths.append(str(path))
    update_prediction_ledger(prediction_root, lottery)
    return EvaluationResult(
        evaluated_count=len(evaluated_paths),
        pending_count=pending_count,
        skipped_count=skipped_count,
        evaluated_paths=tuple(evaluated_paths),
    )


def next_scheduled_draw_date(latest_draw_date: date, lottery: LotteryDefinition) -> date:
    schedule = {day.lower() for day in lottery.draw_schedule}
    candidate = latest_draw_date + timedelta(days=1)
    for _ in range(14):
        if candidate.strftime("%A").lower() in schedule:
            return candidate
        candidate += timedelta(days=1)
    raise ResearchValidationError(f"could not determine next scheduled date for {lottery.code}")


def prediction_lottery_dir(root: str | Path, lottery: LotteryDefinition) -> Path:
    return Path(root) / str(lottery.code)


def prediction_record_path(
    root: str | Path,
    lottery: LotteryDefinition,
    target_draw_number: int,
) -> Path:
    return prediction_lottery_dir(root, lottery) / f"{target_draw_number}.json"


def save_prediction_record(record: PredictionRecord, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_prediction_record(destination)
        if existing.status == PREDICTION_STATUS_EVALUATED:
            raise ResearchValidationError("evaluated prediction records are immutable")
    destination.write_text(research_result_json(record), encoding="utf-8")
    return destination


def load_prediction_record(path: str | Path) -> PredictionRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tickets = tuple(
        PredictionTicket(
            ticket_index=int(ticket["ticket_index"]),
            numbers=tuple(ticket["numbers"]),
            score=float(ticket["score"]),
        )
        for ticket in payload["tickets"]
    )
    return PredictionRecord(
        schema_version=payload["schema_version"],
        status=payload["status"],
        lottery=payload["lottery"],
        target_draw_number=int(payload["target_draw_number"]),
        target_draw_date=payload["target_draw_date"],
        generated_at=payload["generated_at"],
        dataset_hash=payload["dataset_hash"],
        latest_source_draw_number=int(payload["latest_source_draw_number"]),
        latest_source_draw_date=payload["latest_source_draw_date"],
        strategy=payload["strategy"],
        model=payload["model"],
        model_parameters=payload["model_parameters"],
        feature_version=payload["feature_version"],
        feature_group=payload["feature_group"],
        feature_names=tuple(payload["feature_names"]),
        portfolio_method=payload["portfolio_method"],
        portfolio_version=payload["portfolio_version"],
        seed=int(payload["seed"]),
        tickets_per_draw=int(payload["tickets_per_draw"]),
        ticket_price_yen=int(payload["ticket_price_yen"]),
        cost_yen=int(payload["cost_yen"]),
        gross_winnings_yen=None,
        sklearn_version=payload["sklearn_version"],
        config=payload["config"],
        tickets=tickets,
        evaluation=payload["evaluation"],
        warnings=tuple(payload["warnings"]),
    )


def update_prediction_ledger(root: str | Path, lottery: LotteryDefinition) -> Path:
    directory = prediction_lottery_dir(root, lottery)
    directory.mkdir(parents=True, exist_ok=True)
    records = [
        load_prediction_record(path)
        for path in sorted(directory.glob("*.json"))
        if path.name != "ledger.json"
    ]
    payload = {
        "schema_version": "stage10-paper-trading-ledger-v1",
        "lottery": str(lottery.code),
        "entries": [
            {
                "target_draw_number": record.target_draw_number,
                "target_draw_date": record.target_draw_date,
                "prediction_created_before_draw": record.latest_source_draw_number
                < record.target_draw_number,
                "tickets_generated": record.tickets_per_draw,
                "evaluated": record.status == PREDICTION_STATUS_EVALUATED,
                "best_matches": None
                if record.evaluation is None
                else record.evaluation["best_match_count"],
                "prize_categories": ()
                if record.evaluation is None
                else tuple(
                    result["prize_category"] for result in record.evaluation["ticket_results"]
                ),
                "cost_yen": record.cost_yen,
                "gross_winnings_yen": None,
            }
            for record in records
        ],
    }
    path = directory / "ledger.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _score_future_numbers(
    draws: tuple[HistoricalDraw, ...],
    lottery: LotteryDefinition,
    strategy: ProductionStrategyConfig,
    *,
    seed: int,
    target_draw_number: int,
    target_draw_date: date,
) -> dict[int, float]:
    blocks = build_walk_forward_feature_blocks(draws, lottery, strategy.feature_names)
    state = _NumberFeatureState(lottery)
    for draw in draws:
        state.add_draw(draw)
    future_block = type(blocks[0])(
        draw_index=len(blocks),
        draw_number=target_draw_number,
        draw_date=target_draw_date.isoformat(),
        rows=tuple(
            NumberFeatureRow(
                draw_index=len(blocks),
                draw_number=target_draw_number,
                draw_date=target_draw_date.isoformat(),
                number=number,
                features=state.features_for_number(number, strategy.feature_names),
                label=0,
            )
            for number in range(lottery.number_min, lottery.number_max + 1)
        ),
    )
    x_train, y_train, _training_dates = build_training_dataset(
        blocks + (future_block,),
        len(blocks),
    )
    model = _make_model(strategy.model_name, seed)
    model.fit(x_train, y_train)
    return _scores_from_fitted_model(model, future_block)


def _generate_ranked_tickets(
    scores: dict[int, float],
    lottery: LotteryDefinition,
    *,
    tickets_per_draw: int,
    method: str,
    candidate_pool_size: int,
) -> tuple[PredictionTicket, ...]:
    if tickets_per_draw == 2:
        portfolio = construct_portfolio(
            scores,
            lottery,
            method,
            candidate_pool_size=candidate_pool_size,
        )
        return tuple(
            PredictionTicket(index + 1, ticket, portfolio.ticket_scores[index])
            for index, ticket in enumerate(portfolio.tickets)
        )
    ranked = tuple(
        number for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )
    size = lottery.numbers_per_ticket
    required = tickets_per_draw * size
    if len(ranked) < required:
        raise ResearchValidationError("tickets_per_draw exceeds disjoint ranked-ticket capacity")
    tickets = []
    for index in range(tickets_per_draw):
        numbers = lottery.validate_main_numbers(ranked[index * size : (index + 1) * size])
        tickets.append(
            PredictionTicket(
                ticket_index=index + 1,
                numbers=numbers,
                score=sum(scores[number] for number in numbers),
            )
        )
    if len({ticket.numbers for ticket in tickets}) != len(tickets):
        raise ResearchValidationError("generated production tickets must be distinct")
    return tuple(tickets)


def _evaluated_record(
    record: PredictionRecord,
    target_draw: HistoricalDraw,
    lottery: LotteryDefinition,
) -> PredictionRecord:
    if record.status == PREDICTION_STATUS_EVALUATED:
        raise ResearchValidationError("evaluated prediction records are immutable")
    results = []
    for ticket in record.tickets:
        match = match_ticket(ticket.numbers, target_draw, lottery)
        results.append(
            {
                "ticket_index": ticket.ticket_index,
                "numbers": ticket.numbers,
                "main_match_count": match.main_match_count,
                "bonus_match_count": match.bonus_match_count,
                "prize_category": match.prize_name,
                "qualifies_for_prize": match.qualifies_for_prize,
            }
        )
    evaluation = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "actual_draw_number": target_draw.draw_number,
        "actual_draw_date": target_draw.draw_date.isoformat(),
        "actual_main_numbers": target_draw.main_numbers,
        "actual_bonus_numbers": target_draw.bonus_numbers,
        "ticket_results": tuple(results),
        "best_match_count": max(result["main_match_count"] for result in results),
        "prize_qualified_ticket_count": sum(result["qualifies_for_prize"] for result in results),
        "gross_winnings_yen": None,
    }
    return replace(
        record,
        status=PREDICTION_STATUS_EVALUATED,
        evaluation=evaluation,
    )


def _model_parameters(model_name: str, seed: int) -> dict[str, Any]:
    model = _make_model(model_name, seed)
    return model.get_params()


def _record_matches_history(record: PredictionRecord, draws: tuple[HistoricalDraw, ...]) -> bool:
    latest = draws[-1]
    return (
        record.latest_source_draw_number == latest.draw_number
        and record.latest_source_draw_date == latest.draw_date.isoformat()
        and record.dataset_hash == calculate_dataset_hash(draws)
    )
