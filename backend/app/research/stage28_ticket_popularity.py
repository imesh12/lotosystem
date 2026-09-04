from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from backend.app.domain.lottery import LotteryDefinition
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.data import HistoricalDraw, load_draws_csv
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.persistence import research_result_json, to_jsonable
from backend.app.research.settlement import SETTLEMENT_ROOT, load_settlement, settlement_lottery_dir
from backend.app.research.statistical_evaluation import ConfidenceInterval, holm_adjust_p_values

STAGE28_SCHEMA_VERSION = "v2-stage28-ticket-popularity-v1"
STAGE28_EXPERIMENT = "stage28_mini_loto_ticket_popularity"
STAGE28_OUTPUT_DIR = Path("data") / "exports" / "stage28"
MINI_LOTO_COMBINATION_COUNT = 169_911
PRIMARY_ENDPOINT = "popularity_score_vs_sales_normalized_first_winner_rate"
RECOMMENDATION_NONE = "NONE"
RECOMMENDATION_ANTI_POPULARITY = "ANTI_POPULARITY_RESEARCH_CANDIDATE"


@dataclass(frozen=True, slots=True)
class ProxyFeatureDefinition:
    name: str
    definition: str
    rationale: str
    evidence_source: str | None
    classification: str


@dataclass(frozen=True, slots=True)
class PopularityScore:
    ticket: tuple[int, ...]
    components: dict[str, float]
    raw_score: float
    normalized_score: float


@dataclass(frozen=True, slots=True)
class UniverseSummary:
    combination_count: int
    minimum_score: float
    maximum_score: float
    mean_score: float
    median_score: float
    standard_deviation: float
    quantiles: dict[str, float]
    highest_risk_examples: tuple[dict[str, Any], ...]
    lowest_risk_examples: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class HistoricalWinnerScore:
    draw_number: int
    draw_date: str
    main_numbers: tuple[int, ...]
    popularity_score: float
    popularity_percentile: float


@dataclass(frozen=True, slots=True)
class WinnerCountObservation:
    draw_number: int
    draw_date: str
    main_numbers: tuple[int, ...]
    popularity_score: float
    first_prize_winners: int | None
    first_prize_payout_yen: int | None
    sales_amount_yen: int | None
    estimated_tickets_sold: float | None
    normalized_winner_rate: float | None


@dataclass(frozen=True, slots=True)
class AssociationResult:
    endpoint: str
    usable_observations: int
    method: str
    effect: float | None
    confidence_interval: ConfidenceInterval | None
    raw_p_value: float
    holm_p_value: float
    bh_p_value: float
    classification: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class Stage28Result:
    schema_version: str
    experiment: str
    lottery: str
    dataset_hash: str
    draw_count: int
    data_availability: dict[str, Any]
    proxy_feature_definitions: tuple[ProxyFeatureDefinition, ...]
    score_formula: dict[str, Any]
    universe_summary: UniverseSummary
    historical_winning_distribution: dict[str, Any]
    winner_count_observations: tuple[WinnerCountObservation, ...]
    primary_association: AssociationResult
    secondary_associations: dict[str, AssociationResult]
    strongest_individual_component: str | None
    period_segmentation: dict[str, Any]
    conditional_payout_examples: dict[str, Any]
    recommendation: str
    anti_popularity_selector_built: bool
    warnings: tuple[str, ...]


def run_stage28_ticket_popularity_research(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    *,
    settlement_root: str | Path = SETTLEMENT_ROOT,
    output_dir: str | Path | None = STAGE28_OUTPUT_DIR,
    seed: int = 123456,
) -> Stage28Result:
    _require_mini(lottery)
    ordered = validate_lottery_dataset(draws, lottery)
    if not ordered:
        raise ResearchValidationError("Stage 28 requires Mini Loto history")
    universe_scores = score_universe()
    historical = score_historical_winners(ordered, universe_scores)
    observations = winner_count_observations(
        ordered,
        universe_scores,
        settlement_root=settlement_root,
    )
    data_availability = audit_data_availability(
        draws=ordered,
        observations=observations,
        settlement_root=settlement_root,
    )
    association = primary_association_test(observations, seed=seed)
    secondary = component_association_tests(observations, seed=seed)
    recommendation = recommendation_from_association(association)
    result = Stage28Result(
        schema_version=STAGE28_SCHEMA_VERSION,
        experiment=STAGE28_EXPERIMENT,
        lottery=str(lottery.code),
        dataset_hash=calculate_dataset_hash(ordered),
        draw_count=len(ordered),
        data_availability=data_availability,
        proxy_feature_definitions=proxy_feature_definitions(),
        score_formula=score_formula(),
        universe_summary=universe_summary(universe_scores),
        historical_winning_distribution=historical_winning_distribution(historical),
        winner_count_observations=observations,
        primary_association=association,
        secondary_associations=secondary,
        strongest_individual_component=_strongest_component(secondary),
        period_segmentation=period_segmentation(observations),
        conditional_payout_examples=conditional_payout_examples(
            prize_pool_yen=10_000_000,
            split_counts=(1, 2, 3, 5, 10),
        ),
        recommendation=recommendation,
        anti_popularity_selector_built=recommendation == RECOMMENDATION_ANTI_POPULARITY,
        warnings=(
            "Popularity scores are HEURISTIC_POPULARITY_PROXY values, not actual "
            "player-choice frequencies.",
            "Stage 28 does not change draw probability or claim predictive edge.",
            "No production predictions, settlements, or Stage 27 records are modified.",
        ),
    )
    if output_dir is not None:
        save_stage28_outputs(result, output_dir)
    return result


def enumerate_mini_loto_combinations() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(1, 32), 5))


def score_universe() -> tuple[PopularityScore, ...]:
    return tuple(score_ticket(ticket) for ticket in enumerate_mini_loto_combinations())


def score_ticket(ticket: tuple[int, ...] | list[int]) -> PopularityScore:
    numbers = _canonical_ticket(ticket)
    components = popularity_components(numbers)
    raw_score = sum(components.values()) / len(components)
    return PopularityScore(
        ticket=numbers,
        components=components,
        raw_score=raw_score,
        normalized_score=raw_score,
    )


def popularity_components(ticket: tuple[int, ...]) -> dict[str, float]:
    numbers = _canonical_ticket(ticket)
    gaps = tuple(right - left for left, right in zip(numbers, numbers[1:], strict=False))
    even_count = sum(number % 2 == 0 for number in numbers)
    decade_counts = Counter(_decade_bucket(number) for number in numbers)
    digit_counts = Counter(number % 10 for number in numbers)
    mean_sum = 5 * (1 + 31) / 2
    max_sum_distance = max(abs(sum(range(1, 6)) - mean_sum), abs(sum(range(27, 32)) - mean_sum))
    gap_std = pstdev(gaps) if len(gaps) > 1 else 0.0
    return {
        "consecutive_pair_count": _bounded(sum(gap == 1 for gap in gaps) / 4),
        "longest_consecutive_run": _bounded((_longest_run(numbers) - 1) / 4),
        "arithmetic_progression_strength": 1.0 if len(set(gaps)) == 1 else 0.0,
        "low_range_concentration": _bounded((30 - (numbers[-1] - numbers[0])) / 30),
        "low_number_fraction": sum(number <= 15 for number in numbers) / 5,
        "same_decade_concentration": max(decade_counts.values()) / 5,
        "repeated_last_digit_count": _bounded((max(digit_counts.values()) - 1) / 4),
        "even_odd_extremeness": abs(even_count - (5 - even_count)) / 5,
        "sum_extremeness": abs(sum(numbers) - mean_sum) / max_sum_distance,
        "spacing_regularness": _bounded(1 - min(gap_std, 10) / 10),
        "adjacent_gap_repetition": _bounded((Counter(gaps).most_common(1)[0][1] - 1) / 3),
        "simple_sequence_indicator": 1.0
        if numbers == (1, 2, 3, 4, 5) or _longest_run(numbers) >= 4 or len(set(gaps)) == 1
        else 0.0,
        "round_number_fraction": sum(number in {10, 20, 30} for number in numbers) / 5,
    }


def proxy_feature_definitions() -> tuple[ProxyFeatureDefinition, ...]:
    return (
        ProxyFeatureDefinition(
            "consecutive_pair_count",
            "Fraction of adjacent sorted gaps equal to 1.",
            "Sequential tickets are visually simple and commonly suspected human patterns.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "longest_consecutive_run",
            "Longest run of consecutive numbers normalized to 0..1.",
            "Long runs resemble obvious manual patterns.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "arithmetic_progression_strength",
            "1 when all adjacent gaps are identical; otherwise 0.",
            "Even spacing creates memorable pattern-like combinations.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "low_range_concentration",
            "Narrow ticket range transformed so tighter clustering has higher risk.",
            "Clustered selections can look date-like or personally meaningful.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "low_number_fraction",
            "Fraction of selected numbers <= 15.",
            "Lower numbers may be easier date anchors, though Mini Loto numbers all fit "
            "day-of-month values.",
            None,
            "UNSUPPORTED",
        ),
        ProxyFeatureDefinition(
            "same_decade_concentration",
            "Largest decade bucket share among 1-9, 10-19, 20-29, and 30-31.",
            "Same-band clustering is a simple visible pattern.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "repeated_last_digit_count",
            "Maximum repeated final digit count normalized to 0..1.",
            "Repeated endings can be a consciously chosen visual pattern.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "even_odd_extremeness",
            "Distance from balanced even/odd composition.",
            "All-even or all-odd tickets are simple categories.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "sum_extremeness",
            "Distance of ticket sum from the Mini Loto universe mean sum.",
            "Very low or high sums can look less balanced and more pattern-driven.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "spacing_regularness",
            "Higher when adjacent gaps have low standard deviation.",
            "Regular spacing is easy to recognize.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "adjacent_gap_repetition",
            "Normalized count of repeated adjacent gap sizes.",
            "Repeated gaps are another form of visual regularity.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "simple_sequence_indicator",
            "Flag for 1-2-3-4-5, long consecutive runs, or full arithmetic progressions.",
            "Captures the clearest human-pattern sanity-check examples.",
            None,
            "PLAUSIBLE_HEURISTIC",
        ),
        ProxyFeatureDefinition(
            "round_number_fraction",
            "Fraction of 10, 20, and 30 in the ticket.",
            "Round numbers can be salient manual anchors.",
            None,
            "UNSUPPORTED",
        ),
    )


def score_formula() -> dict[str, Any]:
    return {
        "name": "HEURISTIC_POPULARITY_PROXY",
        "direction": "higher score = higher human-pattern popularity risk",
        "normalization": "all components are mapped to 0..1 before averaging",
        "weights": {definition.name: 1.0 for definition in proxy_feature_definitions()},
        "formula": "mean(normalized component values)",
        "interpretation": "heuristic pattern risk only; not actual ticket-selection probability",
    }


def universe_summary(scores: tuple[PopularityScore, ...]) -> UniverseSummary:
    values = tuple(sorted(score.normalized_score for score in scores))
    ranked = sorted(scores, key=lambda score: (score.normalized_score, score.ticket))
    return UniverseSummary(
        combination_count=len(scores),
        minimum_score=values[0],
        maximum_score=values[-1],
        mean_score=mean(values),
        median_score=median(values),
        standard_deviation=pstdev(values),
        quantiles={
            "p01": _quantile(values, 0.01),
            "p05": _quantile(values, 0.05),
            "p25": _quantile(values, 0.25),
            "p75": _quantile(values, 0.75),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
        },
        highest_risk_examples=tuple(_score_example(score) for score in ranked[-10:][::-1]),
        lowest_risk_examples=tuple(_score_example(score) for score in ranked[:10]),
    )


def score_historical_winners(
    draws: tuple[HistoricalDraw, ...],
    universe_scores: tuple[PopularityScore, ...],
) -> tuple[HistoricalWinnerScore, ...]:
    universe_values = tuple(sorted(score.normalized_score for score in universe_scores))
    rows: list[HistoricalWinnerScore] = []
    for draw in draws:
        score = score_ticket(draw.main_numbers)
        rows.append(
            HistoricalWinnerScore(
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                main_numbers=draw.main_numbers,
                popularity_score=score.normalized_score,
                popularity_percentile=_percentile_rank(universe_values, score.normalized_score),
            )
        )
    return tuple(rows)


def historical_winning_distribution(
    rows: tuple[HistoricalWinnerScore, ...],
) -> dict[str, Any]:
    values = tuple(sorted(row.popularity_percentile for row in rows))
    scores = tuple(sorted(row.popularity_score for row in rows))
    return {
        "draw_count": len(rows),
        "mean_percentile": mean(values) if values else None,
        "median_percentile": median(values) if values else None,
        "percentile_quantiles": {
            "p05": _quantile(values, 0.05) if values else None,
            "p25": _quantile(values, 0.25) if values else None,
            "p75": _quantile(values, 0.75) if values else None,
            "p95": _quantile(values, 0.95) if values else None,
        },
        "mean_score": mean(scores) if scores else None,
        "highest_patterned_winners": tuple(
            to_jsonable(row)
            for row in sorted(rows, key=lambda row: row.popularity_score, reverse=True)[:10]
        ),
        "lowest_patterned_winners": tuple(
            to_jsonable(row) for row in sorted(rows, key=lambda row: row.popularity_score)[:10]
        ),
    }


def winner_count_observations(
    draws: tuple[HistoricalDraw, ...],
    universe_scores: tuple[PopularityScore, ...],
    *,
    settlement_root: str | Path = SETTLEMENT_ROOT,
) -> tuple[WinnerCountObservation, ...]:
    draw_by_number = {draw.draw_number: draw for draw in draws}
    universe_values = {score.ticket: score.normalized_score for score in universe_scores}
    observations: list[WinnerCountObservation] = []
    for path in sorted(settlement_lottery_dir(settlement_root, MINI_LOTO).glob("*.json")):
        settlement = load_settlement(path)
        if settlement.draw_number not in draw_by_number:
            continue
        first = next((payout for payout in settlement.payouts if payout.prize_tier == "1st"), None)
        draw = draw_by_number[settlement.draw_number]
        score = universe_values.get(
            draw.main_numbers, score_ticket(draw.main_numbers).normalized_score
        )
        sales_amount = _sales_amount_from_settlement_payload(path)
        tickets_sold = (
            sales_amount / MINI_LOTO.ticket_price_yen if sales_amount is not None else None
        )
        normalized = (
            first.winners_count / tickets_sold
            if first is not None and first.winners_count is not None and tickets_sold
            else None
        )
        observations.append(
            WinnerCountObservation(
                draw_number=draw.draw_number,
                draw_date=draw.draw_date.isoformat(),
                main_numbers=draw.main_numbers,
                popularity_score=score,
                first_prize_winners=None if first is None else first.winners_count,
                first_prize_payout_yen=None if first is None else first.payout_yen,
                sales_amount_yen=sales_amount,
                estimated_tickets_sold=tickets_sold,
                normalized_winner_rate=normalized,
            )
        )
    return tuple(observations)


def audit_data_availability(
    *,
    draws: tuple[HistoricalDraw, ...],
    observations: tuple[WinnerCountObservation, ...],
    settlement_root: str | Path,
) -> dict[str, Any]:
    settlement_dir = settlement_lottery_dir(settlement_root, MINI_LOTO)
    settlement_count = len(tuple(settlement_dir.glob("*.json"))) if settlement_dir.exists() else 0
    return {
        "canonical_mini_loto_draws": len(draws),
        "canonical_contains_winning_numbers": True,
        "actual_purchased_number_distribution": False,
        "winner_ticket_combinations": False,
        "quick_pick_vs_manual_ratio": False,
        "public_popularity_ranking_data": False,
        "settlement_files_found": settlement_count,
        "first_prize_winner_counts_found": sum(
            observation.first_prize_winners is not None for observation in observations
        ),
        "first_prize_payouts_found": sum(
            observation.first_prize_payout_yen is not None for observation in observations
        ),
        "sales_amount_records_found": sum(
            observation.sales_amount_yen is not None for observation in observations
        ),
        "direct_popularity_estimation_possible": False,
        "primary_sales_normalized_association_possible": sum(
            observation.normalized_winner_rate is not None for observation in observations
        )
        >= 5,
        "conclusion": "We cannot directly estimate combination popularity from current data.",
    }


def primary_association_test(
    observations: tuple[WinnerCountObservation, ...],
    *,
    seed: int,
) -> AssociationResult:
    usable = tuple(
        observation
        for observation in observations
        if observation.normalized_winner_rate is not None
    )
    if len(usable) < 5:
        return AssociationResult(
            endpoint=PRIMARY_ENDPOINT,
            usable_observations=len(usable),
            method="spearman(popularity_score, first_prize_winners / estimated_tickets_sold)",
            effect=None,
            confidence_interval=None,
            raw_p_value=1.0,
            holm_p_value=1.0,
            bh_p_value=1.0,
            classification="INCONCLUSIVE",
            reason="sales amount is unavailable or sample size is too small",
        )
    effect = spearman_correlation(
        tuple(observation.popularity_score for observation in usable),
        tuple(float(observation.normalized_winner_rate) for observation in usable),
    )
    raw = permutation_p_value(
        tuple(observation.popularity_score for observation in usable),
        tuple(float(observation.normalized_winner_rate) for observation in usable),
        seed=seed,
        replications=2000,
    )
    holm = holm_adjust_p_values({PRIMARY_ENDPOINT: raw})[PRIMARY_ENDPOINT]
    bh = benjamini_hochberg_adjust_p_values({PRIMARY_ENDPOINT: raw})[PRIMARY_ENDPOINT]
    ci = bootstrap_spearman_ci(usable, seed=seed, replications=1000)
    return AssociationResult(
        endpoint=PRIMARY_ENDPOINT,
        usable_observations=len(usable),
        method="spearman(popularity_score, first_prize_winners / estimated_tickets_sold)",
        effect=effect,
        confidence_interval=ci,
        raw_p_value=raw,
        holm_p_value=holm,
        bh_p_value=bh,
        classification=_classification(effect, holm, ci),
        reason=None,
    )


def component_association_tests(
    observations: tuple[WinnerCountObservation, ...],
    *,
    seed: int,
) -> dict[str, AssociationResult]:
    usable = tuple(
        observation
        for observation in observations
        if observation.normalized_winner_rate is not None
    )
    names = tuple(definition.name for definition in proxy_feature_definitions())
    if len(usable) < 5:
        return {
            name: AssociationResult(
                endpoint=f"{name}_vs_sales_normalized_first_winner_rate",
                usable_observations=len(usable),
                method="spearman(component, first_prize_winners / estimated_tickets_sold)",
                effect=None,
                confidence_interval=None,
                raw_p_value=1.0,
                holm_p_value=1.0,
                bh_p_value=1.0,
                classification="INCONCLUSIVE",
                reason="sales amount is unavailable or sample size is too small",
            )
            for name in names
        }
    raw: dict[str, float] = {}
    effects: dict[str, float] = {}
    for name in names:
        x_values = tuple(
            score_ticket(observation.main_numbers).components[name] for observation in usable
        )
        y_values = tuple(float(observation.normalized_winner_rate) for observation in usable)
        effects[name] = spearman_correlation(x_values, y_values)
        raw[name] = permutation_p_value(
            x_values,
            y_values,
            seed=_derived_seed(seed, name),
            replications=2000,
        )
    holm = holm_adjust_p_values(raw)
    bh = benjamini_hochberg_adjust_p_values(raw)
    return {
        name: AssociationResult(
            endpoint=f"{name}_vs_sales_normalized_first_winner_rate",
            usable_observations=len(usable),
            method="spearman(component, first_prize_winners / estimated_tickets_sold)",
            effect=effects[name],
            confidence_interval=None,
            raw_p_value=raw[name],
            holm_p_value=holm[name],
            bh_p_value=bh[name],
            classification=_classification(effects[name], holm[name], None),
            reason=None,
        )
        for name in names
    }


def bootstrap_spearman_ci(
    observations: tuple[WinnerCountObservation, ...],
    *,
    seed: int,
    replications: int,
) -> ConfidenceInterval:
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replications):
        sample = tuple(rng.choice(observations) for _ in observations)
        estimates.append(
            spearman_correlation(
                tuple(item.popularity_score for item in sample),
                tuple(float(item.normalized_winner_rate) for item in sample),
            )
        )
    ordered = tuple(sorted(estimates))
    return ConfidenceInterval(0.95, _quantile(ordered, 0.025), _quantile(ordered, 0.975))


def spearman_correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ResearchValidationError("spearman inputs must have equal length")
    if len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_den = sum((x - left_mean) ** 2 for x in left_ranks)
    right_den = sum((y - right_mean) ** 2 for y in right_ranks)
    denominator = (left_den * right_den) ** 0.5
    return 0.0 if denominator == 0 else numerator / denominator


def permutation_p_value(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    seed: int,
    replications: int,
) -> float:
    observed = abs(spearman_correlation(left, right))
    rng = random.Random(seed)
    right_values = list(right)
    extreme = 0
    for _ in range(replications):
        rng.shuffle(right_values)
        if abs(spearman_correlation(left, tuple(right_values))) >= observed:
            extreme += 1
    return (extreme + 1) / (replications + 1)


def period_segmentation(observations: tuple[WinnerCountObservation, ...]) -> dict[str, Any]:
    periods = {
        "2010-2014": ("2010-01-01", "2014-12-31"),
        "2015-2019": ("2015-01-01", "2019-12-31"),
        "2020-2023": ("2020-01-01", "2023-12-31"),
        "2024-latest": ("2024-01-01", "9999-12-31"),
    }
    payload: dict[str, Any] = {}
    for label, (start, end) in periods.items():
        usable = tuple(
            observation
            for observation in observations
            if start <= observation.draw_date <= end
            and observation.normalized_winner_rate is not None
        )
        payload[label] = {
            "usable_observations": len(usable),
            "mean_popularity_score": mean(tuple(item.popularity_score for item in usable))
            if usable
            else None,
            "mean_normalized_winner_rate": mean(
                tuple(float(item.normalized_winner_rate) for item in usable)
            )
            if usable
            else None,
        }
    return payload


def recommendation_from_association(association: AssociationResult) -> str:
    if (
        association.classification in {"EVIDENCE", "WEAK_SIGNAL"}
        and association.effect is not None
        and association.effect > 0
    ):
        return RECOMMENDATION_ANTI_POPULARITY
    return RECOMMENDATION_NONE


def conditional_payout_examples(
    *,
    prize_pool_yen: int,
    split_counts: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "assumption": "illustrative fixed prize pool; actual payout formula is not modeled",
        "prize_pool_yen": prize_pool_yen,
        "exact_five_number_hit_probability": f"1/{MINI_LOTO_COMBINATION_COUNT}",
        "examples": {
            str(count): {
                "winner_count": count,
                "conditional_payout_per_winning_ticket_yen": prize_pool_yen // count,
            }
            for count in split_counts
        },
    }


def save_stage28_outputs(result: Stage28Result, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    main_path = destination / "mini_loto_ticket_popularity_report.json"
    compact_path = destination / "mini_loto_popularity_universe_compact.json"
    main_path.write_text(research_result_json(result), encoding="utf-8")
    compact_path.write_text(
        research_result_json(
            {
                "schema_version": STAGE28_SCHEMA_VERSION,
                "lottery": str(MINI_LOTO.code),
                "combination_count": result.universe_summary.combination_count,
                "score_formula": result.score_formula,
                "universe_summary": result.universe_summary,
            }
        ),
        encoding="utf-8",
    )
    return {"report": str(main_path), "universe_compact": str(compact_path)}


def load_default_mini_history() -> tuple[HistoricalDraw, ...]:
    return load_draws_csv(Path("data") / "processed" / "mini_loto_history.csv", MINI_LOTO)


def _canonical_ticket(ticket: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return MINI_LOTO.validate_main_numbers(tuple(ticket))


def _longest_run(numbers: tuple[int, ...]) -> int:
    best = 1
    current = 1
    for left, right in zip(numbers, numbers[1:], strict=False):
        if right == left + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _decade_bucket(number: int) -> str:
    if number < 10:
        return "01-09"
    if number < 20:
        return "10-19"
    if number < 30:
        return "20-29"
    return "30-31"


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _percentile_rank(sorted_values: tuple[float, ...], value: float) -> float:
    lower_or_equal = _bisect_right(sorted_values, value)
    return lower_or_equal / len(sorted_values)


def _bisect_right(values: tuple[float, ...], target: float) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


def _score_example(score: PopularityScore) -> dict[str, Any]:
    return {
        "ticket": score.ticket,
        "score": score.normalized_score,
        "components": score.components,
    }


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end + 1) / 2
        for _, index in ordered[cursor : end + 1]:
            ranks[index] = average_rank
        cursor = end + 1
    return tuple(ranks)


def _sales_amount_from_settlement_payload(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sales_amount_yen", "sales_yen", "draw_sales_yen"):
        value = payload.get(key)
        if value is not None:
            return int(value)
    return None


def _classification(
    effect: float | None,
    holm_p_value: float,
    confidence_interval: ConfidenceInterval | None,
) -> str:
    if effect is None:
        return "INCONCLUSIVE"
    if effect < 0:
        return "NEGATIVE"
    if (
        confidence_interval is not None
        and confidence_interval.lower <= 0 <= confidence_interval.upper
    ):
        return "NO_EVIDENCE"
    if holm_p_value < 0.05:
        return "EVIDENCE"
    if holm_p_value < 0.10:
        return "WEAK_SIGNAL"
    return "NO_EVIDENCE"


def _strongest_component(results: dict[str, AssociationResult]) -> str | None:
    usable = tuple(
        item
        for item in results.items()
        if item[1].effect is not None and item[1].classification != "INCONCLUSIVE"
    )
    if not usable:
        return None
    return max(usable, key=lambda item: abs(float(item[1].effect)))[0]


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return int(digest[:16], 16)


def _require_mini(lottery: LotteryDefinition) -> None:
    if lottery.code != MINI_LOTO.code:
        raise ResearchValidationError("Stage 28 ticket-popularity research supports MINI_LOTO only")
