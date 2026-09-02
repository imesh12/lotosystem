from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
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
from backend.app.research.history_import import canonical_history_path
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.prize import match_ticket
from backend.app.research.production import (
    PREDICTION_ROOT,
    PREDICTION_STATUS_EVALUATED,
    PredictionRecord,
    ProductionStrategyConfig,
    _score_future_numbers,
    load_prediction_record,
    prediction_lottery_dir,
)
from backend.app.research.settlement import (
    SETTLEMENT_ROOT,
    DrawSettlement,
    load_settlement,
    settlement_path,
)

PROSPECTIVE_ROOT = Path("data") / "prospective"
PROSPECTIVE_SCHEMA_VERSION = "stage19-prospective-record-v1"
PROSPECTIVE_SUMMARY_SCHEMA_VERSION = "stage19-prospective-summary-v1"
DEFAULT_PROSPECTIVE_RANDOM_REPLICATIONS = 1000
PROSPECTIVE_TARGET_DRAW_TIME = time(18, 0)
JST = timezone(timedelta(hours=9), "Asia/Tokyo")

DIAGNOSIS_RANKING_WEAK = "RANKING_WEAK"
DIAGNOSIS_PORTFOLIO_WEAK = "PORTFOLIO_WEAK"
DIAGNOSIS_MIXED = "MIXED"
DIAGNOSIS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class WinningNumberRank:
    number: int
    model_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RandomControl:
    replications: int
    seed: int
    tickets_per_draw: int
    mean_best_ticket_matches: float
    mean_total_portfolio_matches: float
    prize_qualified_rate: float
    production_best_match_percentile: float
    production_total_match_percentile: float
    paper_gross_distribution: dict[int, int] | None


@dataclass(frozen=True, slots=True)
class ProspectiveRecord:
    schema_version: str
    lottery: str
    draw_number: int
    draw_date: str
    prediction_id: str
    prediction_path: str
    generated_at: str
    evaluated_at: str
    dataset_hash: str
    model: str
    feature_group: str
    feature_version: str
    feature_names: tuple[str, ...]
    portfolio_method: str
    seed: int
    tickets_per_draw: int
    actual_result: dict[str, Any]
    ticket_results: tuple[dict[str, Any], ...]
    portfolio: dict[str, Any]
    ranking: dict[str, Any]
    random_control: RandomControl
    diagnostic_classification: str
    warnings: tuple[str, ...]


def prospective_evaluate(
    *,
    lottery: LotteryDefinition | None = None,
    prediction_root: str | Path = PREDICTION_ROOT,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    prospective_root: str | Path = PROSPECTIVE_ROOT,
    random_replications: int = DEFAULT_PROSPECTIVE_RANDOM_REPLICATIONS,
) -> dict[str, Any]:
    lotteries = (LOTO6, MINI_LOTO) if lottery is None else (lottery,)
    records: list[ProspectiveRecord] = []
    not_eligible: list[dict[str, Any]] = []
    for selected in lotteries:
        for path in _prediction_paths(prediction_root, selected):
            try:
                record = evaluate_prediction_record(
                    path,
                    selected,
                    settlement_root=settlement_root,
                    prospective_root=prospective_root,
                    random_replications=random_replications,
                )
            except ResearchValidationError as exc:
                if _is_not_eligible(str(exc)):
                    not_eligible.append(
                        {
                            "lottery": str(selected.code),
                            "prediction_path": str(path),
                            "reason": str(exc),
                        }
                    )
                    continue
                raise
            records.append(record)
    summary = prospective_summary(
        lottery=lottery,
        prospective_root=prospective_root,
        save=True,
    )
    return {
        "status": "ok",
        "eligible_records": len(records),
        "not_eligible": tuple(not_eligible),
        "records": tuple(_record_summary(record) for record in records),
        "summary": summary,
    }


def evaluate_prediction_record(
    prediction_path: str | Path,
    lottery: LotteryDefinition,
    *,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    prospective_root: str | Path = PROSPECTIVE_ROOT,
    random_replications: int = DEFAULT_PROSPECTIVE_RANDOM_REPLICATIONS,
) -> ProspectiveRecord:
    record_path = Path(prediction_path)
    prediction = load_prediction_record(record_path)
    _validate_prediction_lottery(prediction, lottery)
    if prediction.status != PREDICTION_STATUS_EVALUATED or prediction.evaluation is None:
        raise ResearchValidationError("NOT_ELIGIBLE: prediction is not evaluated")
    if not _generated_before_target_draw(prediction):
        raise ResearchValidationError("NOT_ELIGIBLE: prediction was not generated before draw")
    settlement = _load_matching_settlement(settlement_root, lottery, prediction)
    training_draws = _training_draws_for_prediction(prediction, lottery)
    scores = _reconstruct_scores(prediction, lottery, training_draws)
    ranking = _ranking_payload(prediction, lottery, scores)
    ticket_results = _ticket_results(prediction, settlement)
    portfolio = _portfolio_payload(prediction, settlement)
    random_control = _random_control(
        prediction,
        lottery,
        settlement,
        seed=prediction.seed if prediction.seed is not None else DEFAULT_STAGE05_SEED,
        replications=random_replications,
    )
    prospective = ProspectiveRecord(
        schema_version=PROSPECTIVE_SCHEMA_VERSION,
        lottery=str(lottery.code),
        draw_number=prediction.target_draw_number,
        draw_date=prediction.target_draw_date,
        prediction_id=record_path.stem,
        prediction_path=str(record_path),
        generated_at=prediction.generated_at,
        evaluated_at=str(prediction.evaluation["evaluated_at"]),
        dataset_hash=prediction.dataset_hash,
        model=prediction.model,
        feature_group=prediction.feature_group,
        feature_version=prediction.feature_version,
        feature_names=prediction.feature_names,
        portfolio_method=prediction.portfolio_method,
        seed=prediction.seed,
        tickets_per_draw=prediction.tickets_per_draw,
        actual_result={
            "main_numbers": tuple(prediction.evaluation["actual_main_numbers"]),
            "bonus_numbers": tuple(prediction.evaluation["actual_bonus_numbers"]),
        },
        ticket_results=ticket_results,
        portfolio=portfolio,
        ranking=ranking,
        random_control=random_control,
        diagnostic_classification=_diagnostic_classification(lottery, ranking, portfolio),
        warnings=(
            "Prospective diagnostics use only predictions saved before result availability.",
            "Ranking diagnostics are research evidence only, not winning probabilities.",
            "Random controls are diagnostic and do not alter production predictions.",
        ),
    )
    save_prospective_record(
        prospective,
        prospective_record_path(prospective_root, lottery, prediction.target_draw_number),
    )
    return prospective


def prospective_summary(
    *,
    lottery: LotteryDefinition | None = None,
    prospective_root: str | Path = PROSPECTIVE_ROOT,
    save: bool = False,
) -> dict[str, Any]:
    lotteries = (LOTO6, MINI_LOTO) if lottery is None else (lottery,)
    by_lottery = {
        str(selected.code): _summary_for_records(
            _load_records(prospective_root, selected),
            selected,
        )
        for selected in lotteries
    }
    payload = {
        "schema_version": PROSPECTIVE_SUMMARY_SCHEMA_VERSION,
        "lotteries": by_lottery,
        "overall_conclusion": _overall_conclusion(tuple(by_lottery.values())),
    }
    if save:
        path = Path(prospective_root) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload["summary_path"] = str(path)
    return payload


def save_prospective_record(record: ProspectiveRecord, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_prospective_record(destination)
        if _prospective_scientific_payload(existing) != _prospective_scientific_payload(record):
            raise ResearchValidationError(
                f"conflicting prospective record for {record.lottery} #{record.draw_number}"
            )
        return destination
    destination.write_text(research_result_json(record), encoding="utf-8")
    return destination


def load_prospective_record(path: str | Path) -> ProspectiveRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProspectiveRecord(
        schema_version=payload["schema_version"],
        lottery=payload["lottery"],
        draw_number=int(payload["draw_number"]),
        draw_date=payload["draw_date"],
        prediction_id=payload["prediction_id"],
        prediction_path=payload["prediction_path"],
        generated_at=payload["generated_at"],
        evaluated_at=payload["evaluated_at"],
        dataset_hash=payload["dataset_hash"],
        model=payload["model"],
        feature_group=payload["feature_group"],
        feature_version=payload["feature_version"],
        feature_names=tuple(payload["feature_names"]),
        portfolio_method=payload["portfolio_method"],
        seed=int(payload["seed"]),
        tickets_per_draw=int(payload["tickets_per_draw"]),
        actual_result=payload["actual_result"],
        ticket_results=tuple(payload["ticket_results"]),
        portfolio=payload["portfolio"],
        ranking=payload["ranking"],
        random_control=RandomControl(**payload["random_control"]),
        diagnostic_classification=payload["diagnostic_classification"],
        warnings=tuple(payload["warnings"]),
    )


def prospective_record_path(
    root: str | Path,
    lottery: LotteryDefinition,
    draw_number: int,
) -> Path:
    return Path(root) / str(lottery.code) / f"{draw_number}.json"


def _validate_prediction_lottery(record: PredictionRecord, lottery: LotteryDefinition) -> None:
    if record.lottery != str(lottery.code):
        raise ResearchValidationError(
            f"prediction lottery {record.lottery} does not match requested {lottery.code}"
        )


def _generated_before_target_draw(record: PredictionRecord) -> bool:
    generated = datetime.fromisoformat(record.generated_at).astimezone(JST)
    target = datetime.combine(
        datetime.fromisoformat(record.target_draw_date).date(),
        PROSPECTIVE_TARGET_DRAW_TIME,
        tzinfo=JST,
    )
    return generated < target


def _load_matching_settlement(
    settlement_root: str | Path,
    lottery: LotteryDefinition,
    prediction: PredictionRecord,
) -> DrawSettlement:
    path = settlement_path(settlement_root, lottery, prediction.target_draw_number)
    if not path.exists():
        raise ResearchValidationError("NOT_ELIGIBLE: settlement is not available")
    settlement = load_settlement(path)
    if settlement.prediction_dataset_hash != prediction.dataset_hash:
        raise ResearchValidationError("settlement dataset hash does not match prediction")
    if settlement.draw_number != prediction.target_draw_number:
        raise ResearchValidationError("settlement draw number does not match prediction")
    return settlement


def _training_draws_for_prediction(
    prediction: PredictionRecord,
    lottery: LotteryDefinition,
) -> tuple[HistoricalDraw, ...]:
    draws = validate_lottery_dataset(
        load_draws_csv(canonical_history_path(lottery), lottery),
        lottery,
    )
    training = tuple(
        draw
        for draw in draws
        if (draw.draw_date, draw.draw_number)
        <= (
            datetime.fromisoformat(prediction.latest_source_draw_date).date(),
            prediction.latest_source_draw_number,
        )
    )
    if not training:
        raise ResearchValidationError("prediction training history is empty")
    latest = training[-1]
    if (
        latest.draw_number != prediction.latest_source_draw_number
        or latest.draw_date.isoformat() != prediction.latest_source_draw_date
    ):
        raise ResearchValidationError("could not reconstruct prediction training boundary")
    if calculate_dataset_hash(training) != prediction.dataset_hash:
        raise ResearchValidationError("could not reconstruct prediction dataset hash exactly")
    return training


def _reconstruct_scores(
    prediction: PredictionRecord,
    lottery: LotteryDefinition,
    training_draws: tuple[HistoricalDraw, ...],
) -> dict[int, float]:
    target_date = datetime.fromisoformat(prediction.target_draw_date).date()
    strategy = ProductionStrategyConfig(
        model_name=prediction.model,
        feature_group=prediction.feature_group,
        feature_names=prediction.feature_names,
        feature_version=prediction.feature_version,
        portfolio_method=prediction.portfolio_method,
        portfolio_version=prediction.portfolio_version,
    )
    return _score_future_numbers(
        training_draws,
        lottery,
        strategy,
        seed=prediction.seed,
        target_draw_number=prediction.target_draw_number,
        target_draw_date=target_date,
    )


def _ranking_payload(
    prediction: PredictionRecord,
    lottery: LotteryDefinition,
    scores: dict[int, float],
) -> dict[str, Any]:
    winners = tuple(prediction.evaluation["actual_main_numbers"]) if prediction.evaluation else ()
    ranked = tuple(
        number for number, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    )
    rank_by_number = {number: index + 1 for index, number in enumerate(ranked)}
    winning_ranks = tuple(
        WinningNumberRank(number, scores[number], rank_by_number[number]) for number in winners
    )
    top_counts = _top_counts(lottery)
    return {
        "winning_number_ranks": winning_ranks,
        "top_numbers": ranked,
        "top_capture": {
            f"top_{top_count}": sum(rank_by_number[number] <= top_count for number in winners)
            for top_count in top_counts
        },
    }


def _top_counts(lottery: LotteryDefinition) -> tuple[int, ...]:
    if str(lottery.code) == "MINI_LOTO":
        return (5, 10, 15, 20)
    return (6, 12, 18, 24)


def _ticket_results(
    prediction: PredictionRecord,
    settlement: DrawSettlement,
) -> tuple[dict[str, Any], ...]:
    payout_by_ticket = {ticket.ticket_index: ticket for ticket in settlement.tickets}
    assert prediction.evaluation is not None
    results = []
    for result in prediction.evaluation["ticket_results"]:
        settled = payout_by_ticket[int(result["ticket_index"])]
        results.append(
            {
                "ticket_index": int(result["ticket_index"]),
                "numbers": tuple(result["numbers"]),
                "main_matches": int(result["main_match_count"]),
                "bonus_match": int(result["bonus_match_count"]),
                "prize_tier": settled.prize_tier,
                "payout_yen": settled.payout_yen,
            }
        )
    return tuple(results)


def _portfolio_payload(prediction: PredictionRecord, settlement: DrawSettlement) -> dict[str, Any]:
    ticket_sets = tuple(set(ticket.numbers) for ticket in prediction.tickets)
    unique_numbers = set().union(*ticket_sets) if ticket_sets else set()
    pair_count = 0
    overlap_total = 0
    for left_index, left in enumerate(ticket_sets):
        for right in ticket_sets[left_index + 1 :]:
            pair_count += 1
            overlap_total += len(left & right)
    assert prediction.evaluation is not None
    matches = tuple(
        int(result["main_match_count"]) for result in prediction.evaluation["ticket_results"]
    )
    return {
        "best_ticket_matches": max(matches) if matches else 0,
        "total_main_matches_across_tickets": sum(matches),
        "unique_numbers_covered": len(unique_numbers),
        "overlap_count": overlap_total,
        "overlap_rate": 0.0
        if pair_count == 0
        else overlap_total / (pair_count * prediction.tickets[0].numbers.__len__()),
        "prize_qualified_tickets": int(prediction.evaluation["prize_qualified_ticket_count"]),
        "paper_cost": settlement.paper_total_cost_yen,
        "paper_gross": settlement.paper_gross_winnings_yen,
        "paper_net": settlement.paper_net_yen,
        "financial_status": settlement.financial_status,
    }


def _random_control(
    prediction: PredictionRecord,
    lottery: LotteryDefinition,
    settlement: DrawSettlement,
    *,
    seed: int,
    replications: int,
) -> RandomControl:
    if replications <= 0:
        raise ResearchValidationError("random replications must be positive")
    draw = HistoricalDraw(
        lottery,
        prediction.target_draw_number,
        datetime.fromisoformat(prediction.target_draw_date).date(),
        tuple(settlement.actual_main_numbers),
        tuple(settlement.actual_bonus_numbers),
    )
    payout_by_tier = {payout.prize_tier: payout.payout_yen for payout in settlement.payouts}
    can_score_gross = bool(payout_by_tier)
    best_matches: list[int] = []
    total_matches: list[int] = []
    prize_portfolios = 0
    gross_values: list[int] = []
    production_best = max(ticket.main_match_count for ticket in settlement.tickets)
    production_total = sum(ticket.main_match_count for ticket in settlement.tickets)
    for replication in range(replications):
        rng = random.Random(seed + prediction.target_draw_number * 10_000 + replication)
        tickets = generate_distinct_random_tickets(lottery, rng, prediction.tickets_per_draw)
        matches = []
        qualifies = False
        gross = 0
        for ticket in tickets:
            result = match_ticket(ticket, draw, lottery)
            matches.append(result.main_match_count)
            qualifies = qualifies or result.qualifies_for_prize
            if can_score_gross and result.prize_name is not None:
                gross += payout_by_tier.get(result.prize_name, 0)
        best_matches.append(max(matches))
        total_matches.append(sum(matches))
        prize_portfolios += int(qualifies)
        if can_score_gross:
            gross_values.append(gross)
    return RandomControl(
        replications=replications,
        seed=seed,
        tickets_per_draw=prediction.tickets_per_draw,
        mean_best_ticket_matches=mean(best_matches),
        mean_total_portfolio_matches=mean(total_matches),
        prize_qualified_rate=prize_portfolios / replications,
        production_best_match_percentile=_percentile_at_or_below(best_matches, production_best),
        production_total_match_percentile=_percentile_at_or_below(total_matches, production_total),
        paper_gross_distribution=None
        if not can_score_gross
        else dict(sorted(Counter(gross_values).items())),
    )


def _diagnostic_classification(
    lottery: LotteryDefinition,
    ranking: dict[str, Any],
    portfolio: dict[str, Any],
) -> str:
    capture = ranking["top_capture"]
    top_region = f"top_{lottery.numbers_per_ticket * 2}"
    broad_region = f"top_{lottery.numbers_per_ticket * 4}"
    top_capture = int(capture.get(top_region, 0))
    broad_capture = int(capture.get(broad_region, 0))
    best_ticket = int(portfolio["best_ticket_matches"])
    if broad_capture <= 1:
        return DIAGNOSIS_RANKING_WEAK
    if top_capture >= lottery.numbers_per_ticket // 2 and best_ticket < top_capture:
        return DIAGNOSIS_PORTFOLIO_WEAK
    if broad_capture >= 2 and best_ticket <= 2:
        return DIAGNOSIS_MIXED
    return DIAGNOSIS_INSUFFICIENT_DATA


def _summary_for_records(
    records: tuple[ProspectiveRecord, ...],
    lottery: LotteryDefinition,
) -> dict[str, Any]:
    if not records:
        return {
            "lottery": str(lottery.code),
            "eligible_draws": 0,
            "evaluated_draws": 0,
            "conclusion": DIAGNOSIS_INSUFFICIENT_DATA,
        }
    total_tickets = sum(record.tickets_per_draw for record in records)
    production_total = tuple(
        record.portfolio["total_main_matches_across_tickets"] for record in records
    )
    random_total = tuple(record.random_control.mean_total_portfolio_matches for record in records)
    cumulative = []
    running_production = 0.0
    running_random = 0.0
    for index, record in enumerate(records, start=1):
        running_production += record.portfolio["total_main_matches_across_tickets"]
        running_random += record.random_control.mean_total_portfolio_matches
        cumulative.append(
            {
                "draw_number": record.draw_number,
                "production_average_total_matches": running_production / index,
                "random_average_total_matches": running_random / index,
            }
        )
    complete_financial = all(record.portfolio["paper_gross"] is not None for record in records)
    return {
        "lottery": str(lottery.code),
        "eligible_draws": len(records),
        "evaluated_draws": len(records),
        "total_tickets": total_tickets,
        "production_average_best_ticket_matches": mean(
            tuple(record.portfolio["best_ticket_matches"] for record in records)
        ),
        "production_average_total_portfolio_matches": mean(production_total),
        "production_prize_qualified_rate": sum(
            record.portfolio["prize_qualified_tickets"] > 0 for record in records
        )
        / len(records),
        "production_paper_cost": sum(record.portfolio["paper_cost"] for record in records),
        "production_paper_gross": None
        if not complete_financial
        else sum(record.portfolio["paper_gross"] or 0 for record in records),
        "production_paper_net": None
        if not complete_financial
        else sum(record.portfolio["paper_net"] or 0 for record in records),
        "random_mean_total_portfolio_matches": mean(random_total),
        "production_minus_random_total_matches": mean(
            tuple(left - right for left, right in zip(production_total, random_total, strict=True))
        ),
        "production_percentile": mean(
            tuple(record.random_control.production_total_match_percentile for record in records)
        ),
        "cumulative_by_draw": tuple(cumulative),
        "rolling_windows": {
            "trailing_5": _rolling(records, 5),
            "trailing_10": _rolling(records, 10),
        },
        "conclusion": DIAGNOSIS_INSUFFICIENT_DATA
        if len(records) < 5
        else "prospective_tracking_active",
    }


def _rolling(records: tuple[ProspectiveRecord, ...], window: int) -> tuple[dict[str, Any], ...]:
    rows = []
    for index in range(window, len(records) + 1):
        selected = records[index - window : index]
        rows.append(
            {
                "ending_draw_number": selected[-1].draw_number,
                "production_average_total_matches": mean(
                    tuple(
                        record.portfolio["total_main_matches_across_tickets"] for record in selected
                    )
                ),
                "random_average_total_matches": mean(
                    tuple(record.random_control.mean_total_portfolio_matches for record in selected)
                ),
            }
        )
    return tuple(rows)


def _load_records(
    prospective_root: str | Path,
    lottery: LotteryDefinition,
) -> tuple[ProspectiveRecord, ...]:
    directory = Path(prospective_root) / str(lottery.code)
    if not directory.exists():
        return ()
    return tuple(load_prospective_record(path) for path in sorted(directory.glob("*.json")))


def _prediction_paths(
    prediction_root: str | Path,
    lottery: LotteryDefinition,
) -> tuple[Path, ...]:
    directory = prediction_lottery_dir(prediction_root, lottery)
    if not directory.exists():
        return ()
    return tuple(path for path in sorted(directory.glob("*.json")) if path.name != "ledger.json")


def _record_summary(record: ProspectiveRecord) -> dict[str, Any]:
    return {
        "lottery": record.lottery,
        "draw_number": record.draw_number,
        "draw_date": record.draw_date,
        "diagnostic_classification": record.diagnostic_classification,
        "best_ticket_matches": record.portfolio["best_ticket_matches"],
        "total_portfolio_matches": record.portfolio["total_main_matches_across_tickets"],
        "top_capture": record.ranking["top_capture"],
    }


def _prospective_scientific_payload(record: ProspectiveRecord) -> dict[str, Any]:
    return to_jsonable(record)


def _is_not_eligible(message: str) -> bool:
    return message.startswith("NOT_ELIGIBLE:")


def _overall_conclusion(summaries: tuple[dict[str, Any], ...]) -> str:
    if not summaries or all(summary.get("eligible_draws", 0) < 5 for summary in summaries):
        return DIAGNOSIS_INSUFFICIENT_DATA
    return "prospective_tracking_active"


def _percentile_at_or_below(values: list[int], observed: int) -> float:
    return sum(value <= observed for value in values) / len(values) if values else 0.0
