"""Stage 30 — Mini Loto exact ticket portfolio optimization.

Stage 30 answers a single, narrow, non-predictive question:

    Given a fixed number of Mini Loto tickets per draw, and a FAIR
    (uniform-random) draw, which ticket-construction structure maximizes
    the EXACT probability of winning at least one prize?

This is pure combinatorics / coverage-design optimization. It does not
discover, assume, or rely on any predictive signal. Stage 29 established
that no historical predictive hypothesis in this repository survives
authoritative global correction; Stage 30 takes that as given and asks a
different question entirely.

Hard safety rules (see AGENTS.md and the Stage 30 task brief):
    * This module NEVER mutates production predictions, Stage 27 records,
      canonical history, settlements, or any other repository state. Every
      function here is a pure, read-only computation over explicit inputs
      or over files it only *reads*.
    * This module never claims predictive improvement. Scenario 2
      ("model-weighted diagnostic") is explicitly and permanently labeled
      as a hypothetical, non-authoritative diagnostic.

Mathematical method
--------------------
For a fixed set of N tickets (each a 5-number subset of 1..31), classify
every number 1..31 by exactly which tickets contain it. This partitions the
31-number universe into a small number of disjoint "regions" (at most
2**N - 1 non-empty regions, usually far fewer). A uniformly random 5-number
winning draw is equivalent to a draw from a multivariate hypergeometric
distribution over those regions: the number of ways the draw can place
exactly `a_r` of its 5 numbers into region `r` (for every region
simultaneously) is the product of C(region_size_r, a_r), and dividing by
C(31, 5) gives the exact probability of that specific allocation.

Because the draw size is only 5, the number of valid allocations
("weak compositions of 5 across the regions, respecting each region's
capacity") is always small (at most a few hundred to a few thousand even
for 6 overlapping tickets), so every portfolio can be scored EXACTLY, with
no Monte Carlo and no need to enumerate all C(31,5) = 169,911 raw draws.

Each ticket's main-match count is the sum of the allocation over the
regions it owns. Mini Loto's bonus-dependent tiers (2nd vs 3rd prize, both
at 4 main matches) are resolved exactly too: a ticket with a deficiency of
exactly 1 (4 of its 5 numbers matched) has its single "missing" number
pinned to one specific region (the unique region where the allocation is
exactly one short of that region's full size). Two tickets' missing numbers
coincide if and only if they are pinned to the *same* region, because a
region is a single group of interchangeable-but-shared numbers. This lets
Stage 30 compute the probability that the one real bonus number upgrades a
ticket from 3rd to 2nd prize, exactly, via simple counting, without ever
enumerating the identity of the bonus number explicitly.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

from backend.app.domain.lottery import LotteryDefinition, PrizeTier
from backend.app.domain.rules import MINI_LOTO
from backend.app.research.data import load_draws_csv, validate_draw_sequence

STAGE30_SCHEMA_VERSION = "v2-stage30-ticket-portfolio-optimization-v1"
DEFAULT_STAGE30_OUTPUT_DIR = Path("data") / "exports" / "stage30"
MINI_LOTO_CANONICAL_HISTORY = Path("data") / "processed" / "mini_loto_history.csv"
MINI_LOTO_PRODUCTION_PREDICTION = Path("data") / "predictions" / "MINI_LOTO" / "1404.json"

DIAGNOSTIC_DISCLAIMER = (
    "DIAGNOSTIC ONLY. This scenario treats production model scores as if they were "
    "genuine winning probabilities. Stage 29's global research ledger found that no "
    "predictive hypothesis in this repository survives authoritative correction, so "
    "this scenario is NOT evidence of real increased winning probability. It exists "
    "only to show how the optimal portfolio structure would differ IF the scores were "
    "real, which is explicitly not established."
)


# ---------------------------------------------------------------------------
# A. Authoritative rule verification (never hardcode tier semantics)
# ---------------------------------------------------------------------------


def build_qualify_table(lottery: LotteryDefinition) -> dict[tuple[int, bool], PrizeTier | None]:
    """Reproduce backend.app.research.prize.match_ticket's tier-selection logic
    as a lookup table keyed by (main_match_count, bonus_matched), verified
    directly from the lottery's own authoritative prize_tiers -- nothing here
    is assumed independently of that definition."""
    table: dict[tuple[int, bool], PrizeTier | None] = {}
    tiers_by_rank = sorted(lottery.prize_tiers, key=lambda tier: tier.rank)
    for main_match_count in range(0, lottery.numbers_per_ticket + 1):
        for bonus_matched in (False, True):
            selected: PrizeTier | None = None
            for tier in tiers_by_rank:
                if main_match_count != tier.required_main_matches:
                    continue
                if tier.requires_bonus and not bonus_matched:
                    continue
                selected = tier
                break
            table[(main_match_count, bonus_matched)] = selected
    return table


def minimum_prize_match_count(lottery: LotteryDefinition) -> int:
    """The lowest main-match count that can ever qualify for a prize,
    verified against the authoritative tiers (not assumed)."""
    return min(tier.required_main_matches for tier in lottery.prize_tiers)


def deterministic_qualifying_match_counts(lottery: LotteryDefinition) -> frozenset[int]:
    """Match counts that qualify for some prize even in the worst case
    (bonus not matched). Verified from the authoritative tiers."""
    table = build_qualify_table(lottery)
    return frozenset(
        count
        for count in range(0, lottery.numbers_per_ticket + 1)
        if table[(count, False)] is not None
    )


def bonus_sensitive_match_counts(lottery: LotteryDefinition) -> frozenset[int]:
    """Match counts where the achieved tier genuinely depends on whether the
    bonus number is also in the ticket (i.e. the tier differs between
    bonus_matched=True and bonus_matched=False). Verified, not assumed."""
    table = build_qualify_table(lottery)
    sensitive = set()
    for count in range(0, lottery.numbers_per_ticket + 1):
        if table[(count, False)] is not table[(count, True)]:
            sensitive.add(count)
    return frozenset(sensitive)


def assert_monotonic_bonus_eligibility(lottery: LotteryDefinition) -> None:
    """Stage 30's probability shortcuts rely on one verified structural fact:
    the bonus number can only ever UPGRADE a ticket's tier, never create
    eligibility from ineligibility, and never remove eligibility. This
    function checks that fact against the authoritative tiers and raises if
    a future rule change breaks the assumption, instead of silently
    producing wrong numbers."""
    table = build_qualify_table(lottery)
    for count in range(0, lottery.numbers_per_ticket + 1):
        no_bonus = table[(count, False)]
        with_bonus = table[(count, True)]
        if no_bonus is None and with_bonus is not None:
            raise AssertionError(
                "Stage 30 assumption violated: bonus creates eligibility from "
                f"ineligibility at main_match_count={count}. The closed-form "
                "deterministic shortcuts in this module are not valid for this "
                "rule set."
            )
        if no_bonus is not None and with_bonus is not None and with_bonus.rank > no_bonus.rank:
            raise AssertionError(
                "Stage 30 assumption violated: bonus makes the tier worse at "
                f"main_match_count={count}. The closed-form deterministic "
                "shortcuts in this module are not valid for this rule set."
            )


@dataclass(frozen=True, slots=True)
class RuleVerification:
    lottery_name: str
    universe_min: int
    universe_max: int
    win_size: int
    bonus_numbers: int
    tiers: tuple[dict[str, Any], ...]
    minimum_prize_match_count: int
    deterministic_qualifying_match_counts: tuple[int, ...]
    bonus_sensitive_match_counts: tuple[int, ...]
    monotonic_bonus_eligibility_verified: bool


def verify_rules(lottery: LotteryDefinition = MINI_LOTO) -> RuleVerification:
    assert_monotonic_bonus_eligibility(lottery)
    tiers = tuple(
        {
            "rank": tier.rank,
            "name": tier.name,
            "required_main_matches": tier.required_main_matches,
            "requires_bonus": bool(tier.requires_bonus),
            "theoretical_payout_yen": tier.theoretical_payout_yen,
            "odds": tier.odds,
        }
        for tier in sorted(lottery.prize_tiers, key=lambda tier: tier.rank)
    )
    return RuleVerification(
        lottery_name=lottery.name,
        universe_min=lottery.number_min,
        universe_max=lottery.number_max,
        win_size=lottery.numbers_per_ticket,
        bonus_numbers=lottery.bonus_numbers,
        tiers=tiers,
        minimum_prize_match_count=minimum_prize_match_count(lottery),
        deterministic_qualifying_match_counts=tuple(
            sorted(deterministic_qualifying_match_counts(lottery))
        ),
        bonus_sensitive_match_counts=tuple(sorted(bonus_sensitive_match_counts(lottery))),
        monotonic_bonus_eligibility_verified=True,
    )


# ---------------------------------------------------------------------------
# B. Exact outcome space
# ---------------------------------------------------------------------------


def exact_outcome_space_size(lottery: LotteryDefinition = MINI_LOTO) -> int:
    return comb(lottery.number_max, lottery.numbers_per_ticket)


MINI_LOTO_TOTAL_OUTCOMES = exact_outcome_space_size(MINI_LOTO)
assert MINI_LOTO_TOTAL_OUTCOMES == 169_911, "Mini Loto outcome-space size changed unexpectedly"


# ---------------------------------------------------------------------------
# C/D/E. Exact region-based portfolio evaluation engine
# ---------------------------------------------------------------------------


def region_decomposition(
    tickets: Sequence[Sequence[int]], universe: int = MINI_LOTO.number_max
) -> tuple[tuple[frozenset[int], ...], tuple[int, ...]]:
    """Partition the universe into regions by exactly which tickets own each
    number. Works for any number of tickets N. Returns (owner_sets, sizes)
    with zero-size regions already dropped."""
    ticket_sets = [frozenset(ticket) for ticket in tickets]
    counts: dict[frozenset[int], int] = {}
    for number in range(1, universe + 1):
        owners = frozenset(i for i, ticket in enumerate(ticket_sets) if number in ticket)
        counts[owners] = counts.get(owners, 0) + 1
    owner_sets = tuple(counts.keys())
    sizes = tuple(counts.values())
    return owner_sets, sizes


def _weak_compositions_with_capacity(
    total: int, capacities: Sequence[int]
) -> list[tuple[int, ...]]:
    """All ways to split `total` indistinguishable draw-slots across regions
    with capacities `capacities[r]`, respecting 0 <= a_r <= capacities[r]."""
    n = len(capacities)
    suffix_capacity = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_capacity[i] = suffix_capacity[i + 1] + capacities[i]
    if total > suffix_capacity[0]:
        return []

    results: list[tuple[int, ...]] = []

    def recurse(index: int, remaining: int, chosen: list[int]) -> None:
        if index == n:
            if remaining == 0:
                results.append(tuple(chosen))
            return
        max_here = min(capacities[index], remaining)
        min_here = max(0, remaining - suffix_capacity[index + 1])
        for value in range(min_here, max_here + 1):
            chosen.append(value)
            recurse(index + 1, remaining - value, chosen)
            chosen.pop()

    recurse(0, total, [])
    return results


@dataclass(frozen=True, slots=True)
class PortfolioOutcome:
    schema_version: str
    n_tickets: int
    tickets: tuple[tuple[int, ...], ...] | None
    distinct_numbers: int
    region_sizes: tuple[int, ...]
    total_outcomes: int
    p_at_least_one_prize: Fraction
    p_any_4th_prize: Fraction
    p_3rd_or_better: Fraction
    p_2nd_or_better: Fraction
    p_1st_prize: Fraction
    expected_winning_tickets_per_draw: Fraction

    def to_jsonable(self) -> dict[str, Any]:
        def frac(value: Fraction) -> dict[str, Any]:
            return {
                "numerator": value.numerator,
                "denominator": value.denominator,
                "float": float(value),
            }

        return {
            "schema_version": self.schema_version,
            "n_tickets": self.n_tickets,
            "tickets": [list(ticket) for ticket in self.tickets] if self.tickets else None,
            "distinct_numbers": self.distinct_numbers,
            "region_sizes": list(self.region_sizes),
            "total_outcomes": self.total_outcomes,
            "p_at_least_one_prize": frac(self.p_at_least_one_prize),
            "p_any_4th_prize": frac(self.p_any_4th_prize),
            "p_3rd_or_better": frac(self.p_3rd_or_better),
            "p_2nd_or_better": frac(self.p_2nd_or_better),
            "p_1st_prize": frac(self.p_1st_prize),
            "expected_winning_tickets_per_draw": frac(self.expected_winning_tickets_per_draw),
        }


def evaluate_region_sizes(
    owner_sets: Sequence[frozenset[int]],
    sizes: Sequence[int],
    n_tickets: int,
    lottery: LotteryDefinition = MINI_LOTO,
    tickets: tuple[tuple[int, ...], ...] | None = None,
) -> PortfolioOutcome:
    """The core exact engine. `owner_sets[r]`/`sizes[r]` describe region r:
    which ticket indices (0..n_tickets-1) own it, and how many numbers are in
    it. Must cover the full universe (sizes sum to lottery.number_max)."""
    assert_monotonic_bonus_eligibility(lottery)
    win_size = lottery.numbers_per_ticket
    universe = lottery.number_max
    bonus_pool = universe - win_size
    total_outcomes = comb(universe, win_size)
    if sum(sizes) != universe:
        raise ValueError(f"region sizes must sum to universe size {universe}, got {sum(sizes)}")

    active = [(owners, size) for owners, size in zip(owner_sets, sizes, strict=True) if size > 0]
    if not active:
        raise ValueError("no active regions")
    active_owner_sets = [owners for owners, _ in active]
    active_sizes = [size for _, size in active]

    qualify_table = build_qualify_table(lottery)
    deterministic_matches = deterministic_qualifying_match_counts(lottery)
    bonus_sensitive_matches = bonus_sensitive_match_counts(lottery)
    rank_by_match_count_no_bonus = {
        count: (qualify_table[(count, False)].rank if qualify_table[(count, False)] else None)
        for count in range(0, win_size + 1)
    }
    rank_by_match_count_with_bonus = {
        count: (qualify_table[(count, True)].rank if qualify_table[(count, True)] else None)
        for count in range(0, win_size + 1)
    }
    rank_4th = max(tier.rank for tier in lottery.prize_tiers)  # worst/minimum qualifying tier
    rank_3rd_or_better_cutoff = sorted({tier.rank for tier in lottery.prize_tiers})[-2]
    rank_1st = min(tier.rank for tier in lottery.prize_tiers)

    weak_compositions = _weak_compositions_with_capacity(win_size, active_sizes)

    total_weight_checked = 0
    weight_any_prize = 0
    weight_any_4th = 0
    weight_3rd_or_better = 0
    weight_1st = 0
    weight_2nd_or_better_numerator = 0  # scaled by bonus_pool
    expected_winning_tickets_numerator = 0

    owned_regions_by_ticket: list[list[int]] = [[] for _ in range(n_tickets)]
    for region_index, owners in enumerate(active_owner_sets):
        for ticket_index in owners:
            owned_regions_by_ticket[ticket_index].append(region_index)

    for allocation in weak_compositions:
        weight = 1
        for size, drawn in zip(active_sizes, allocation, strict=True):
            weight *= comb(size, drawn)
        if weight == 0:
            continue
        total_weight_checked += weight

        match_counts = [0] * n_tickets
        for region_index, owners in enumerate(active_owner_sets):
            drawn = allocation[region_index]
            if drawn == 0:
                continue
            for ticket_index in owners:
                match_counts[ticket_index] += drawn

        any_prize = any(count in deterministic_matches for count in match_counts)
        if any_prize:
            weight_any_prize += weight

        any_4th = any(rank_by_match_count_no_bonus.get(count) == rank_4th for count in match_counts)
        if any_4th:
            weight_any_4th += weight

        deterministic_3rd_or_better = any(
            rank_by_match_count_no_bonus.get(count) is not None
            and rank_by_match_count_no_bonus[count] <= rank_3rd_or_better_cutoff
            for count in match_counts
        )
        if deterministic_3rd_or_better:
            weight_3rd_or_better += weight

        any_1st = any(rank_by_match_count_no_bonus.get(count) == rank_1st for count in match_counts)
        if any_1st:
            weight_1st += weight

        expected_winning_tickets_numerator += weight * sum(
            1 for count in match_counts if count in deterministic_matches
        )

        # 2nd-or-better: deterministic via rank-1 tickets, plus bonus-dependent
        # upgrade candidates at bonus-sensitive match counts with deficiency 1.
        deterministic_2nd_or_better = any(
            rank_by_match_count_no_bonus.get(count) is not None
            and rank_by_match_count_no_bonus[count] <= 2
            for count in match_counts
        )
        if deterministic_2nd_or_better:
            weight_2nd_or_better_numerator += weight * bonus_pool
        else:
            maybe_tickets = [
                ticket_index
                for ticket_index, count in enumerate(match_counts)
                if count in bonus_sensitive_matches
                and rank_by_match_count_with_bonus.get(count) is not None
                and rank_by_match_count_with_bonus[count] <= 2
            ]
            if maybe_tickets:
                signature_regions: set[int] = set()
                for ticket_index in maybe_tickets:
                    deficient = [
                        region_index
                        for region_index in owned_regions_by_ticket[ticket_index]
                        if allocation[region_index] == active_sizes[region_index] - 1
                    ]
                    saturated = [
                        region_index
                        for region_index in owned_regions_by_ticket[ticket_index]
                        if allocation[region_index] == active_sizes[region_index]
                    ]
                    if len(deficient) != 1 or len(deficient) + len(saturated) != len(
                        owned_regions_by_ticket[ticket_index]
                    ):
                        raise AssertionError(
                            "expected exactly one deficient region for a ticket with "
                            "deficiency 1; Stage 30's bonus-upgrade shortcut requires "
                            "ticket_size - match_count == 1 at bonus-sensitive counts"
                        )
                    signature_regions.add(deficient[0])
                weight_2nd_or_better_numerator += weight * len(signature_regions)

    if total_weight_checked != total_outcomes:
        raise AssertionError(
            f"weak-composition enumeration did not cover all outcomes: "
            f"{total_weight_checked} != {total_outcomes}"
        )

    distinct_numbers = sum(active_sizes[i] for i, owners in enumerate(active_owner_sets) if owners)

    return PortfolioOutcome(
        schema_version=STAGE30_SCHEMA_VERSION,
        n_tickets=n_tickets,
        tickets=tickets,
        distinct_numbers=distinct_numbers,
        region_sizes=tuple(sizes),
        total_outcomes=total_outcomes,
        p_at_least_one_prize=Fraction(weight_any_prize, total_outcomes),
        p_any_4th_prize=Fraction(weight_any_4th, total_outcomes),
        p_3rd_or_better=Fraction(weight_3rd_or_better, total_outcomes),
        p_2nd_or_better=Fraction(weight_2nd_or_better_numerator, bonus_pool * total_outcomes),
        p_1st_prize=Fraction(weight_1st, total_outcomes),
        expected_winning_tickets_per_draw=Fraction(
            expected_winning_tickets_numerator, total_outcomes
        ),
    )


def evaluate_portfolio_exact(
    tickets: Sequence[Sequence[int]], lottery: LotteryDefinition = MINI_LOTO
) -> PortfolioOutcome:
    owner_sets, sizes = region_decomposition(tickets, universe=lottery.number_max)
    concrete = tuple(tuple(sorted(ticket)) for ticket in tickets)
    return evaluate_region_sizes(
        owner_sets, sizes, n_tickets=len(tickets), lottery=lottery, tickets=concrete
    )


# ---------------------------------------------------------------------------
# E. Intersection-structure global search (N = 3, exhaustive, exact)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThreeTicketStructure:
    i_ab: int
    i_ac: int
    i_bc: int
    i_abc: int

    def region_sizes(self, ticket_size: int = 5, universe: int = 31) -> dict[str, int] | None:
        only_a = ticket_size - self.i_ab - self.i_ac + self.i_abc
        only_b = ticket_size - self.i_ab - self.i_bc + self.i_abc
        only_c = ticket_size - self.i_ac - self.i_bc + self.i_abc
        ab_only = self.i_ab - self.i_abc
        ac_only = self.i_ac - self.i_abc
        bc_only = self.i_bc - self.i_abc
        abc = self.i_abc
        used = only_a + only_b + only_c + ab_only + ac_only + bc_only + abc
        outside = universe - used
        values = {
            "only_a": only_a,
            "only_b": only_b,
            "only_c": only_c,
            "ab_only": ab_only,
            "ac_only": ac_only,
            "bc_only": bc_only,
            "abc": abc,
            "outside": outside,
        }
        if any(v < 0 for v in values.values()):
            return None
        return values


def enumerate_three_ticket_structures(
    ticket_size: int = 5,
) -> Iterable[ThreeTicketStructure]:
    for i_ab in range(0, ticket_size + 1):
        for i_ac in range(0, ticket_size + 1):
            for i_bc in range(0, ticket_size + 1):
                max_abc = min(i_ab, i_ac, i_bc)
                for i_abc in range(0, max_abc + 1):
                    structure = ThreeTicketStructure(i_ab, i_ac, i_bc, i_abc)
                    if structure.region_sizes(ticket_size) is not None:
                        yield structure


def _owner_sizes_from_three_ticket_structure(
    structure: ThreeTicketStructure, ticket_size: int, universe: int
) -> tuple[tuple[frozenset[int], ...], tuple[int, ...]]:
    regions = structure.region_sizes(ticket_size, universe)
    if regions is None:
        raise ValueError("invalid structure")
    owner_sets = (
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 2}),
        frozenset({0, 1, 2}),
        frozenset(),
    )
    sizes = (
        regions["only_a"],
        regions["only_b"],
        regions["only_c"],
        regions["ab_only"],
        regions["ac_only"],
        regions["bc_only"],
        regions["abc"],
        regions["outside"],
    )
    return owner_sets, sizes


def evaluate_three_ticket_structure(
    structure: ThreeTicketStructure, lottery: LotteryDefinition = MINI_LOTO
) -> PortfolioOutcome:
    owner_sets, sizes = _owner_sizes_from_three_ticket_structure(
        structure, lottery.numbers_per_ticket, lottery.number_max
    )
    return evaluate_region_sizes(owner_sets, sizes, n_tickets=3, lottery=lottery, tickets=None)


def materialize_three_ticket_structure(
    structure: ThreeTicketStructure, lottery: LotteryDefinition = MINI_LOTO
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build one concrete, valid set of real Mini Loto numbers realizing the
    given intersection structure, for reporting and cross-validation."""
    regions = structure.region_sizes(lottery.numbers_per_ticket, lottery.number_max)
    if regions is None:
        raise ValueError("invalid structure")
    cursor = 1
    region_numbers: dict[str, list[int]] = {}
    for name in ("only_a", "only_b", "only_c", "ab_only", "ac_only", "bc_only", "abc", "outside"):
        count = regions[name]
        region_numbers[name] = list(range(cursor, cursor + count))
        cursor += count
    ticket_a = sorted(
        region_numbers["only_a"]
        + region_numbers["ab_only"]
        + region_numbers["ac_only"]
        + region_numbers["abc"]
    )
    ticket_b = sorted(
        region_numbers["only_b"]
        + region_numbers["ab_only"]
        + region_numbers["bc_only"]
        + region_numbers["abc"]
    )
    ticket_c = sorted(
        region_numbers["only_c"]
        + region_numbers["ac_only"]
        + region_numbers["bc_only"]
        + region_numbers["abc"]
    )
    return tuple(ticket_a), tuple(ticket_b), tuple(ticket_c)


@dataclass(frozen=True, slots=True)
class GlobalSearchResult:
    lottery_name: str
    n_tickets: int
    structures_evaluated: int
    best_structure: dict[str, int]
    best_p_at_least_one_prize: Fraction
    best_tickets_example: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    disjoint_structure: dict[str, int]
    disjoint_p_at_least_one_prize: Fraction
    disjoint_is_optimal: bool
    full_table: tuple[dict[str, Any], ...]


def search_optimal_three_ticket_portfolio(
    lottery: LotteryDefinition = MINI_LOTO,
) -> GlobalSearchResult:
    ticket_size = lottery.numbers_per_ticket
    best: tuple[Fraction, ThreeTicketStructure, PortfolioOutcome] | None = None
    table: list[dict[str, Any]] = []
    evaluated = 0
    disjoint_structure = ThreeTicketStructure(0, 0, 0, 0)
    disjoint_outcome: PortfolioOutcome | None = None

    for structure in enumerate_three_ticket_structures(ticket_size):
        outcome = evaluate_three_ticket_structure(structure, lottery)
        evaluated += 1
        table.append(
            {
                "i_ab": structure.i_ab,
                "i_ac": structure.i_ac,
                "i_bc": structure.i_bc,
                "i_abc": structure.i_abc,
                "distinct_numbers": outcome.distinct_numbers,
                "p_at_least_one_prize": float(outcome.p_at_least_one_prize),
            }
        )
        if structure == disjoint_structure:
            disjoint_outcome = outcome
        if best is None or outcome.p_at_least_one_prize > best[0]:
            best = (outcome.p_at_least_one_prize, structure, outcome)

    if best is None or disjoint_outcome is None:
        raise RuntimeError("global search produced no valid structures")

    best_p, best_structure, best_outcome = best
    best_tickets = materialize_three_ticket_structure(best_structure, lottery)

    table.sort(key=lambda row: row["p_at_least_one_prize"], reverse=True)

    return GlobalSearchResult(
        lottery_name=lottery.name,
        n_tickets=3,
        structures_evaluated=evaluated,
        best_structure={
            "i_ab": best_structure.i_ab,
            "i_ac": best_structure.i_ac,
            "i_bc": best_structure.i_bc,
            "i_abc": best_structure.i_abc,
        },
        best_p_at_least_one_prize=best_p,
        best_tickets_example=best_tickets,
        disjoint_structure={"i_ab": 0, "i_ac": 0, "i_bc": 0, "i_abc": 0},
        disjoint_p_at_least_one_prize=disjoint_outcome.p_at_least_one_prize,
        disjoint_is_optimal=(best_p == disjoint_outcome.p_at_least_one_prize),
        full_table=tuple(table),
    )


# ---------------------------------------------------------------------------
# C. Baseline portfolios + random-sample comparison
# ---------------------------------------------------------------------------


def disjoint_tickets(
    n_tickets: int, ticket_size: int = 5, universe: int = 31
) -> tuple[tuple[int, ...], ...]:
    max_tickets = universe // ticket_size
    if n_tickets > max_tickets:
        raise ValueError(
            f"cannot build {n_tickets} fully disjoint {ticket_size}-tickets from {universe} numbers"
        )
    tickets = []
    cursor = 1
    for _ in range(n_tickets):
        tickets.append(tuple(range(cursor, cursor + ticket_size)))
        cursor += ticket_size
    return tuple(tickets)


def identical_tickets(n_tickets: int, ticket_size: int = 5) -> tuple[tuple[int, ...], ...]:
    base = tuple(range(1, ticket_size + 1))
    return tuple(base for _ in range(n_tickets))


def overlap_pool_tickets(
    pool_size: int, n_tickets: int = 3, ticket_size: int = 5
) -> tuple[tuple[int, ...], ...]:
    """Build n_tickets tickets of size ticket_size, all drawn from a reduced
    pool of `pool_size` distinct numbers (1..pool_size), spreading overlap as
    evenly as possible so every pool number is used by at least one ticket
    and ticket fill counts stay balanced. Used for the explicit
    7..15-unique-number baselines. Deterministic, no randomness."""
    return overlap_pool_tickets_from_numbers(list(range(1, pool_size + 1)), n_tickets, ticket_size)


def baseline_portfolios(
    lottery: LotteryDefinition = MINI_LOTO,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    ticket_size = lottery.numbers_per_ticket
    universe = lottery.number_max
    portfolios: dict[str, tuple[tuple[int, ...], ...]] = {
        "1_ticket": disjoint_tickets(1, ticket_size, universe),
        "2_disjoint_tickets": disjoint_tickets(2, ticket_size, universe),
        "3_disjoint_tickets": disjoint_tickets(3, ticket_size, universe),
        "3_identical_tickets": identical_tickets(3, ticket_size),
    }
    for pool_size in range(ticket_size + 2, 3 * ticket_size + 1):  # 7..15 for ticket_size=5
        portfolios[f"3_tickets_pool_{pool_size}"] = overlap_pool_tickets(pool_size, 3, ticket_size)
    return portfolios


def evaluate_baseline_portfolios(
    lottery: LotteryDefinition = MINI_LOTO,
) -> dict[str, dict[str, Any]]:
    results = {}
    for name, tickets in baseline_portfolios(lottery).items():
        outcome = evaluate_portfolio_exact(tickets, lottery)
        results[name] = outcome.to_jsonable()
    return results


def deterministic_random_portfolio_sample(
    n_samples: int,
    seed: int,
    n_tickets: int = 3,
    lottery: LotteryDefinition = MINI_LOTO,
) -> dict[str, Any]:
    rng = random.Random(seed)
    universe = list(range(1, lottery.number_max + 1))
    ticket_size = lottery.numbers_per_ticket
    best_value = Fraction(-1)
    best_tickets: tuple[tuple[int, ...], ...] | None = None
    worst_value = Fraction(2)
    worst_tickets: tuple[tuple[int, ...], ...] | None = None
    values: list[float] = []
    for _ in range(n_samples):
        tickets = tuple(tuple(sorted(rng.sample(universe, ticket_size))) for _ in range(n_tickets))
        outcome = evaluate_portfolio_exact(tickets, lottery)
        values.append(float(outcome.p_at_least_one_prize))
        if outcome.p_at_least_one_prize > best_value:
            best_value = outcome.p_at_least_one_prize
            best_tickets = tickets
        if outcome.p_at_least_one_prize < worst_value:
            worst_value = outcome.p_at_least_one_prize
            worst_tickets = tickets
    values.sort()
    mean_value = sum(values) / len(values)
    return {
        "seed": seed,
        "n_samples": n_samples,
        "n_tickets": n_tickets,
        "min_p_at_least_one_prize": values[0],
        "max_p_at_least_one_prize": values[-1],
        "mean_p_at_least_one_prize": mean_value,
        "median_p_at_least_one_prize": values[len(values) // 2],
        "best_tickets_found": [list(t) for t in best_tickets] if best_tickets else None,
        "best_p_at_least_one_prize": float(best_value),
        "worst_tickets_found": [list(t) for t in worst_tickets] if worst_tickets else None,
        "worst_p_at_least_one_prize": float(worst_value),
    }


# ---------------------------------------------------------------------------
# F. Comparison against actual current production (#1404) -- read-only
# ---------------------------------------------------------------------------


def load_current_production_tickets(path: Path = MINI_LOTO_PRODUCTION_PREDICTION) -> dict[str, Any]:
    """Read-only. Never writes to this path. Returns the raw tickets plus
    bookkeeping fields so the caller can confirm immutability was respected."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tickets = tuple(tuple(sorted(entry["numbers"])) for entry in data["tickets"])
    return {
        "source_path": str(path),
        "lottery": data["lottery"],
        "target_draw_number": data["target_draw_number"],
        "status": data["status"],
        "tickets": tickets,
    }


def compare_production_to_optimum(
    production_path: Path = MINI_LOTO_PRODUCTION_PREDICTION,
    lottery: LotteryDefinition = MINI_LOTO,
) -> dict[str, Any]:
    production = load_current_production_tickets(production_path)
    production_outcome = evaluate_portfolio_exact(production["tickets"], lottery)
    search = search_optimal_three_ticket_portfolio(lottery)
    return {
        "production": {
            "target_draw_number": production["target_draw_number"],
            "status": production["status"],
            "tickets": [list(t) for t in production["tickets"]],
            **production_outcome.to_jsonable(),
        },
        "global_optimum": {
            "structure": search.best_structure,
            "example_tickets": [list(t) for t in search.best_tickets_example],
            "p_at_least_one_prize": float(search.best_p_at_least_one_prize),
        },
        "production_is_globally_optimal_structure": (
            production_outcome.p_at_least_one_prize == search.best_p_at_least_one_prize
        ),
        "absolute_difference": float(
            search.best_p_at_least_one_prize - production_outcome.p_at_least_one_prize
        ),
    }


# ---------------------------------------------------------------------------
# H. Cost / long-run interpretation
# ---------------------------------------------------------------------------


def long_run_horizon_probabilities(
    p_at_least_one_prize: Fraction, horizons: Sequence[int] = (10, 20, 50)
) -> dict[str, Any]:
    p = float(p_at_least_one_prize)
    q = 1.0 - p
    results = {}
    for horizon in horizons:
        prob_zero = q**horizon
        prob_at_least_one = 1.0 - prob_zero
        results[str(horizon)] = {
            "draws": horizon,
            "p_zero_prize_events": prob_zero,
            "p_at_least_one_prize_event": prob_at_least_one,
        }
    expected_draws_between_hits = (1.0 / p) if p > 0 else float("inf")
    return {
        "p_per_draw": p,
        "expected_draws_between_prize_events": expected_draws_between_hits,
        "horizons": results,
        "disclaimer": (
            "These are long-run averages of a fair random process, not a guarantee "
            "of periodic wins. A streak of many consecutive non-winning draws remains "
            "fully possible even at the optimal portfolio structure."
        ),
    }


# ---------------------------------------------------------------------------
# I. Ticket-count sensitivity (1..6), representative not exhaustive
# ---------------------------------------------------------------------------


def ticket_count_sensitivity(
    max_tickets: int = 6, lottery: LotteryDefinition = MINI_LOTO
) -> dict[str, Any]:
    ticket_size = lottery.numbers_per_ticket
    universe = lottery.number_max
    results = {}
    for n in range(1, max_tickets + 1):
        row: dict[str, Any] = {}
        disjoint = (
            disjoint_tickets(n, ticket_size, universe) if n * ticket_size <= universe else None
        )
        if disjoint is not None:
            outcome = evaluate_portfolio_exact(disjoint, lottery)
            row["disjoint"] = outcome.to_jsonable()
        else:
            row["disjoint"] = None
        identical = identical_tickets(n, ticket_size)
        row["identical"] = evaluate_portfolio_exact(identical, lottery).to_jsonable()
        if disjoint is not None and n >= 2:
            # one representative mild-overlap alternative: a pool 2 tickets'
            # worth smaller than fully disjoint, for comparison
            pool_size = min(universe, n * ticket_size - (n - 1))
            overlap = overlap_pool_tickets(pool_size, n, ticket_size)
            row["mild_overlap"] = evaluate_portfolio_exact(overlap, lottery).to_jsonable()
            row["disjoint_is_best_of_tested"] = (
                disjoint is not None
                and row["disjoint"]["p_at_least_one_prize"]["float"]
                >= row["mild_overlap"]["p_at_least_one_prize"]["float"]
            )
        results[str(n)] = row
    return results


# ---------------------------------------------------------------------------
# G. Fair-null vs model-weighted diagnostic
# ---------------------------------------------------------------------------


def fair_null_optimum(lottery: LotteryDefinition = MINI_LOTO) -> dict[str, Any]:
    search = search_optimal_three_ticket_portfolio(lottery)
    return {
        "label": "SCENARIO_1_FAIR_LOTTERY_NULL",
        "description": "All numbers equally likely; mathematically optimal 3-ticket structure.",
        "best_structure": search.best_structure,
        "example_tickets": [list(t) for t in search.best_tickets_example],
        "p_at_least_one_prize": float(search.best_p_at_least_one_prize),
        "disjoint_is_optimal": search.disjoint_is_optimal,
    }


def _recent_frequency_weights(
    history_path: Path, lottery: LotteryDefinition, cutoff_draw: int, lookback: int = 20
) -> dict[int, int]:
    draws = load_draws_csv(history_path, lottery)
    validate_draw_sequence(draws)
    history = [draw for draw in draws if draw.draw_number <= cutoff_draw]
    window = history[-lookback:]
    counts = {number: 0 for number in range(lottery.number_min, lottery.number_max + 1)}
    for draw in window:
        for number in draw.main_numbers:
            counts[number] += 1
    return counts


def model_weighted_diagnostic(
    history_path: Path = MINI_LOTO_CANONICAL_HISTORY,
    production_path: Path = MINI_LOTO_PRODUCTION_PREDICTION,
    lottery: LotteryDefinition = MINI_LOTO,
    n_simulations: int = 200_000,
    seed: int = 20300001,
) -> dict[str, Any]:
    """SCENARIO 2 -- explicitly diagnostic, not authoritative. Uses the same
    historical feature family as current Mini Loto production
    (recent-frequency / pair_strength_rate's rescaling) to assign weights to
    each number, then asks: IF this weighting were a real probability model,
    would the disjoint ticket structure still be best? Uses a seeded weighted
    Monte Carlo simulation (clearly labeled, not presented as exact) because
    exact weighted-without-replacement enumeration (Wallenius' distribution)
    is not needed to answer this secondary, explicitly hypothetical
    question."""
    production = load_current_production_tickets(production_path)
    data = json.loads(production_path.read_text(encoding="utf-8"))
    cutoff_draw_number = data["latest_source_draw_number"]

    weights = _recent_frequency_weights(history_path, lottery, cutoff_draw_number, lookback=20)
    numbers = sorted(weights.keys())
    weight_values = [max(weights[n], 1) for n in numbers]  # avoid zero-weight exclusion

    rng = random.Random(seed)

    def weighted_draw() -> set[int]:
        pool = list(numbers)
        pool_weights = list(weight_values)
        drawn: set[int] = set()
        for _ in range(lottery.numbers_per_ticket):
            total = sum(pool_weights)
            r = rng.uniform(0, total)
            acc = 0.0
            chosen_index = len(pool) - 1
            for index, weight in enumerate(pool_weights):
                acc += weight
                if r <= acc:
                    chosen_index = index
                    break
            drawn.add(pool.pop(chosen_index))
            pool_weights.pop(chosen_index)
        return drawn

    candidates = {
        "disjoint_top15": disjoint_tickets(3, lottery.numbers_per_ticket, lottery.number_max),
        "production_actual": production["tickets"],
        "pool_9_overlap": overlap_pool_tickets(9, 3, lottery.numbers_per_ticket),
    }
    # Re-map the abstract "disjoint_top15" structural example onto the
    # model's actual top-15 ranked numbers, so the diagnostic reflects the
    # real ranking, not arbitrary labels 1..15.
    ranked_numbers = sorted(numbers, key=lambda n: (-weights[n], n))
    top15 = ranked_numbers[:15]
    candidates["disjoint_top15"] = (
        tuple(sorted(top15[0:5])),
        tuple(sorted(top15[5:10])),
        tuple(sorted(top15[10:15])),
    )
    top9 = ranked_numbers[:9]
    candidates["pool_9_overlap"] = overlap_pool_tickets_from_numbers(
        top9, 3, lottery.numbers_per_ticket
    )

    results: dict[str, Any] = {}
    deterministic_matches = deterministic_qualifying_match_counts(lottery)
    for name, tickets in candidates.items():
        ticket_sets = [set(t) for t in tickets]
        hits = 0
        for _ in range(n_simulations):
            draw = weighted_draw()
            if any(len(draw & ticket) in deterministic_matches for ticket in ticket_sets):
                hits += 1
        results[name] = {
            "tickets": [sorted(t) for t in tickets],
            "simulated_p_at_least_one_prize": hits / n_simulations,
            "n_simulations": n_simulations,
        }

    return {
        "label": "SCENARIO_2_MODEL_WEIGHTED_DIAGNOSTIC",
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "weighting_source": (
            "recent 20-draw main-number frequency, same family as production's pair_only feature"
        ),
        "cutoff_draw_number": cutoff_draw_number,
        "candidates": results,
        "method": (
            "seeded weighted-without-replacement Monte Carlo (not exact; explicitly diagnostic)"
        ),
        "seed": seed,
    }


def overlap_pool_tickets_from_numbers(
    pool: Sequence[int], n_tickets: int, ticket_size: int
) -> tuple[tuple[int, ...], ...]:
    """Deterministically build n_tickets tickets of size ticket_size drawn
    from `pool`, such that every pool element is used by at least one ticket
    (if the pool is small enough to require overlap) and ticket fill counts
    are kept balanced throughout. No randomness; no retry loop that can
    fail to converge.

    Algorithm: total slot budget is n_tickets * ticket_size. Distribute that
    budget across the pool as evenly as possible (each element used
    floor(budget/pool_size) or that +1 times, capped at n_tickets since a
    ticket cannot contain the same number twice). For each element, greedily
    assign its occurrences to whichever currently-least-filled tickets do
    not already contain it.
    """
    pool = list(pool)
    pool_size = len(pool)
    total_slots = n_tickets * ticket_size
    if pool_size > total_slots:
        raise ValueError(
            f"pool of {pool_size} numbers cannot all be used across only {total_slots} ticket slots"
        )
    base, remainder = divmod(total_slots, pool_size)
    if base > n_tickets or (base == n_tickets and remainder > 0):
        raise ValueError(
            f"cannot spread pool of {pool_size} across {n_tickets} tickets "
            f"of size {ticket_size} without repeating a number within one ticket"
        )

    multiplicities = [base + 1 if i < remainder else base for i in range(pool_size)]

    tickets: list[list[int]] = [[] for _ in range(n_tickets)]
    fill_counts = [0] * n_tickets
    for number, multiplicity in zip(pool, multiplicities, strict=True):
        if multiplicity == 0:
            continue
        candidate_order = sorted(range(n_tickets), key=lambda t: (fill_counts[t], t))
        chosen = [t for t in candidate_order if len(tickets[t]) < ticket_size][:multiplicity]
        if len(chosen) < multiplicity:
            raise RuntimeError("overlap_pool_tickets_from_numbers: infeasible allocation")
        for ticket_index in chosen:
            tickets[ticket_index].append(number)
            fill_counts[ticket_index] += 1

    if any(len(t) != ticket_size for t in tickets):
        raise RuntimeError("overlap_pool_tickets_from_numbers: uneven result")
    return tuple(tuple(sorted(t)) for t in tickets)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_stage30(
    repo_root: Path,
    random_sample_size: int = 4000,
    random_seed: int = 300030,
    run_diagnostic: bool = True,
    diagnostic_simulations: int = 200_000,
) -> dict[str, Any]:
    history_path = repo_root / MINI_LOTO_CANONICAL_HISTORY
    production_path = repo_root / MINI_LOTO_PRODUCTION_PREDICTION

    rules = verify_rules(MINI_LOTO)
    search = search_optimal_three_ticket_portfolio(MINI_LOTO)
    baselines = evaluate_baseline_portfolios(MINI_LOTO)
    random_sample = deterministic_random_portfolio_sample(
        random_sample_size, random_seed, 3, MINI_LOTO
    )
    comparison = compare_production_to_optimum(production_path, MINI_LOTO)
    horizons = long_run_horizon_probabilities(search.best_p_at_least_one_prize, (10, 20, 50))
    sensitivity = ticket_count_sensitivity(6, MINI_LOTO)
    scenario1 = fair_null_optimum(MINI_LOTO)
    scenario2 = (
        model_weighted_diagnostic(
            history_path, production_path, MINI_LOTO, n_simulations=diagnostic_simulations
        )
        if run_diagnostic
        else None
    )

    result = {
        "schema_version": STAGE30_SCHEMA_VERSION,
        "rules_verification": dict(
            lottery_name=rules.lottery_name,
            universe_min=rules.universe_min,
            universe_max=rules.universe_max,
            win_size=rules.win_size,
            bonus_numbers=rules.bonus_numbers,
            tiers=list(rules.tiers),
            minimum_prize_match_count=rules.minimum_prize_match_count,
            deterministic_qualifying_match_counts=list(rules.deterministic_qualifying_match_counts),
            bonus_sensitive_match_counts=list(rules.bonus_sensitive_match_counts),
            monotonic_bonus_eligibility_verified=rules.monotonic_bonus_eligibility_verified,
        ),
        "exact_outcome_space_size": exact_outcome_space_size(MINI_LOTO),
        "baseline_portfolios": baselines,
        "global_search": {
            "structures_evaluated": search.structures_evaluated,
            "best_structure": search.best_structure,
            "best_p_at_least_one_prize": float(search.best_p_at_least_one_prize),
            "best_tickets_example": [list(t) for t in search.best_tickets_example],
            "disjoint_p_at_least_one_prize": float(search.disjoint_p_at_least_one_prize),
            "disjoint_is_optimal": search.disjoint_is_optimal,
            "full_table_top_20": search.full_table[:20],
            "full_table_bottom_5": search.full_table[-5:],
        },
        "random_portfolio_sample": random_sample,
        "production_comparison": comparison,
        "long_run_horizons": horizons,
        "ticket_count_sensitivity": sensitivity,
        "scenario_1_fair_null": scenario1,
        "scenario_2_model_weighted_diagnostic": scenario2,
    }
    return result


def save_stage30_outputs(
    result: dict[str, Any], output_dir: Path = DEFAULT_STAGE30_OUTPUT_DIR
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "stage30_mini_loto_portfolio_optimization.json"
    full_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    summary = {
        "schema_version": result["schema_version"],
        "best_structure": result["global_search"]["best_structure"],
        "best_p_at_least_one_prize": result["global_search"]["best_p_at_least_one_prize"],
        "disjoint_is_optimal": result["global_search"]["disjoint_is_optimal"],
        "production_is_globally_optimal_structure": result["production_comparison"][
            "production_is_globally_optimal_structure"
        ],
    }
    summary_path = output_dir / "stage30_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"full": str(full_path), "summary": str(summary_path)}
