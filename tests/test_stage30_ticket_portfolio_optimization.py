from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from backend.app.domain.rules import MINI_LOTO
from backend.app.research.data import HistoricalDraw
from backend.app.research.prize import match_ticket
from backend.app.research.stage30_ticket_portfolio_optimization import (
    MINI_LOTO_PRODUCTION_PREDICTION,
    ThreeTicketStructure,
    build_qualify_table,
    compare_production_to_optimum,
    deterministic_qualifying_match_counts,
    disjoint_tickets,
    enumerate_three_ticket_structures,
    evaluate_portfolio_exact,
    evaluate_three_ticket_structure,
    exact_outcome_space_size,
    identical_tickets,
    load_current_production_tickets,
    materialize_three_ticket_structure,
    minimum_prize_match_count,
    model_weighted_diagnostic,
    overlap_pool_tickets,
    region_decomposition,
    search_optimal_three_ticket_portfolio,
    ticket_count_sensitivity,
    verify_rules,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# A. Exact outcome space
# ---------------------------------------------------------------------------


def test_outcome_space_is_exactly_169911() -> None:
    assert exact_outcome_space_size(MINI_LOTO) == 169_911
    assert comb(31, 5) == 169_911


def test_minimum_prize_match_count_is_verified_not_assumed() -> None:
    # Mini Loto's authoritative 4th tier requires 3 main matches with no
    # bonus condition -- verified from backend.app.domain.rules.MINI_LOTO,
    # not hardcoded independently.
    assert minimum_prize_match_count(MINI_LOTO) == 3
    assert deterministic_qualifying_match_counts(MINI_LOTO) == frozenset({3, 4, 5})


def test_rule_verification_matches_authoritative_tiers() -> None:
    verification = verify_rules(MINI_LOTO)
    ranks = {tier["rank"]: tier for tier in verification.tiers}
    assert ranks[1]["required_main_matches"] == 5
    assert ranks[1]["requires_bonus"] is False
    assert ranks[2]["required_main_matches"] == 4
    assert ranks[2]["requires_bonus"] is True
    assert ranks[3]["required_main_matches"] == 4
    assert ranks[3]["requires_bonus"] is False
    assert ranks[4]["required_main_matches"] == 3
    assert ranks[4]["requires_bonus"] is False
    assert verification.bonus_sensitive_match_counts == (4,)
    assert verification.monotonic_bonus_eligibility_verified is True


# ---------------------------------------------------------------------------
# B. Exact prize classification cross-validated against the repository's
#    own authoritative backend.app.research.prize.match_ticket
# ---------------------------------------------------------------------------


def _draw(main_numbers: tuple[int, ...], bonus_numbers: tuple[int, ...] = ()) -> HistoricalDraw:
    from datetime import date

    return HistoricalDraw(
        lottery=MINI_LOTO,
        draw_number=1,
        draw_date=date(2020, 1, 1),
        main_numbers=main_numbers,
        bonus_numbers=bonus_numbers,
    )


@pytest.mark.parametrize(
    "ticket,main,bonus,expected_name",
    [
        ((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (6,), "1st"),
        ((1, 2, 3, 4, 6), (1, 2, 3, 4, 5), (6,), "2nd"),
        ((1, 2, 3, 4, 7), (1, 2, 3, 4, 5), (6,), "3rd"),
        ((1, 2, 3, 8, 9), (1, 2, 3, 4, 5), (6,), "4th"),
        ((1, 2, 10, 11, 12), (1, 2, 3, 4, 5), (6,), None),
    ],
)
def test_qualify_table_matches_authoritative_match_ticket(
    ticket: tuple[int, ...],
    main: tuple[int, ...],
    bonus: tuple[int, ...],
    expected_name: str | None,
) -> None:
    draw = _draw(main, bonus)
    authoritative = match_ticket(ticket, draw, MINI_LOTO)
    assert authoritative.prize_name == expected_name

    table = build_qualify_table(MINI_LOTO)
    bonus_matched = len(set(ticket) & set(bonus)) > 0
    looked_up = table[(authoritative.main_match_count, bonus_matched)]
    looked_up_name = looked_up.name if looked_up else None
    assert looked_up_name == expected_name


def test_bonus_number_never_coincides_with_main_numbers_in_real_history() -> None:
    from backend.app.research.data import load_draws_csv, validate_draw_sequence

    draws = load_draws_csv(REPO_ROOT / "data" / "processed" / "mini_loto_history.csv", MINI_LOTO)
    validate_draw_sequence(draws)
    for draw in draws:
        assert set(draw.main_numbers).isdisjoint(set(draw.bonus_numbers))


# ---------------------------------------------------------------------------
# C/D. Exact portfolio evaluation: known baselines
# ---------------------------------------------------------------------------


def test_disjoint_three_tickets_known_exact_probability() -> None:
    tickets = disjoint_tickets(3)
    outcome = evaluate_portfolio_exact(tickets)
    assert outcome.p_at_least_one_prize == Fraction(161, 2697)
    assert outcome.distinct_numbers == 15
    # For disjoint tickets, at most one ticket can ever win (pigeonhole: a
    # single draw has only 5 winning numbers, and 2 disjoint 3+-match
    # tickets would require at least 6), so "at least one wins" and
    # "expected winning tickets" must coincide exactly.
    assert outcome.p_at_least_one_prize == outcome.expected_winning_tickets_per_draw


def test_identical_tickets_reduce_to_single_ticket_probability() -> None:
    identical = identical_tickets(3)
    outcome = evaluate_portfolio_exact(identical)
    single_ticket_p = Fraction(
        comb(5, 3) * comb(26, 2) + comb(5, 4) * comb(26, 1) + comb(5, 5) * comb(26, 0),
        comb(31, 5),
    )
    assert outcome.p_at_least_one_prize == single_ticket_p
    assert outcome.distinct_numbers == 5


def test_single_ticket_matches_hand_derived_hypergeometric_value() -> None:
    outcome = evaluate_portfolio_exact(disjoint_tickets(1))
    assert outcome.p_at_least_one_prize == Fraction(3381, 169_911)


def test_portfolio_union_probability_is_monotonic_in_distinct_numbers() -> None:
    # Across the 7..15 unique-number overlap baselines, more distinct
    # numbers must never decrease the probability of at least one prize.
    previous = None
    for pool_size in range(7, 16):
        tickets = overlap_pool_tickets(pool_size, 3, 5)
        outcome = evaluate_portfolio_exact(tickets)
        assert outcome.distinct_numbers == pool_size
        if previous is not None:
            assert outcome.p_at_least_one_prize >= previous
        previous = outcome.p_at_least_one_prize


def test_cumulative_tiers_are_ordered_and_bounded() -> None:
    outcome = evaluate_portfolio_exact(disjoint_tickets(3))
    assert 0 <= outcome.p_1st_prize <= outcome.p_2nd_or_better <= outcome.p_3rd_or_better
    assert outcome.p_3rd_or_better <= outcome.p_at_least_one_prize
    assert outcome.p_any_4th_prize <= outcome.p_at_least_one_prize


# ---------------------------------------------------------------------------
# E. Intersection-structure equivalence and deterministic global search
# ---------------------------------------------------------------------------


def test_intersection_structure_equivalence_under_relabeling() -> None:
    # Two concrete ticket triples realizing the same (i_ab, i_ac, i_bc,
    # i_abc) structure, built from completely different number ranges,
    # must produce the exact same probabilities (pure symmetry argument).
    structure = ThreeTicketStructure(i_ab=1, i_ac=1, i_bc=0, i_abc=0)
    tickets_a = materialize_three_ticket_structure(structure, MINI_LOTO)
    # Shift every number by 10 (still within 1..31 given the small region
    # sizes here) to get a structurally identical but numerically disjoint
    # realization, then evaluate both independently.
    shift = 10
    tickets_b = tuple(
        tuple(sorted(n + shift if n + shift <= 31 else n for n in t)) for t in tickets_a
    )
    outcome_a = evaluate_portfolio_exact(tickets_a)
    outcome_b = evaluate_portfolio_exact(tickets_b)
    outcome_structure = evaluate_three_ticket_structure(structure, MINI_LOTO)
    assert outcome_a.p_at_least_one_prize == outcome_structure.p_at_least_one_prize
    assert outcome_b.p_at_least_one_prize == outcome_structure.p_at_least_one_prize
    assert outcome_a.distinct_numbers == outcome_structure.distinct_numbers
    assert outcome_b.distinct_numbers == outcome_structure.distinct_numbers


def test_weak_composition_enumeration_is_exhaustive() -> None:
    # evaluate_region_sizes asserts internally that composition weights sum
    # to the full outcome space; calling it at all over a nontrivial
    # structure is itself the exhaustiveness check (it would raise
    # AssertionError otherwise).
    structure = ThreeTicketStructure(i_ab=2, i_ac=1, i_bc=1, i_abc=1)
    outcome = evaluate_three_ticket_structure(structure, MINI_LOTO)
    assert outcome.total_outcomes == 169_911


def test_global_search_is_deterministic() -> None:
    first = search_optimal_three_ticket_portfolio(MINI_LOTO)
    second = search_optimal_three_ticket_portfolio(MINI_LOTO)
    assert first.best_structure == second.best_structure
    assert first.best_p_at_least_one_prize == second.best_p_at_least_one_prize
    assert first.structures_evaluated == second.structures_evaluated


def test_global_search_confirms_disjoint_is_optimal_for_mini_loto() -> None:
    result = search_optimal_three_ticket_portfolio(MINI_LOTO)
    assert result.disjoint_is_optimal is True
    assert result.best_structure == {"i_ab": 0, "i_ac": 0, "i_bc": 0, "i_abc": 0}
    assert result.best_p_at_least_one_prize == Fraction(161, 2697)
    # Confirm it is a genuine maximum over the full evaluated table, not
    # just a locally-best candidate.
    best_from_table = max(row["p_at_least_one_prize"] for row in result.full_table)
    assert best_from_table == pytest.approx(float(result.best_p_at_least_one_prize))


def test_global_search_covers_all_valid_structures_exhaustively() -> None:
    structures = list(enumerate_three_ticket_structures(5))
    assert len(structures) > 100  # sanity: genuinely exhaustive, not a handful of examples
    result = search_optimal_three_ticket_portfolio(MINI_LOTO)
    assert result.structures_evaluated == len(structures)


def test_fair_null_optimization_takes_no_history_input() -> None:
    # The fair-null global search's public signature accepts only a lottery
    # definition -- no draw history, no dataset path, nothing that could
    # leak future or historical outcome information into the "optimal
    # structure under uniform randomness" computation.
    import inspect

    signature = inspect.signature(search_optimal_three_ticket_portfolio)
    for name in signature.parameters:
        assert name in {"lottery"}


# ---------------------------------------------------------------------------
# F. Production comparison -- read-only, no mutation
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_1404_tickets_match_global_optimum_exactly() -> None:
    comparison = compare_production_to_optimum(
        REPO_ROOT / MINI_LOTO_PRODUCTION_PREDICTION, MINI_LOTO
    )
    assert comparison["production"]["target_draw_number"] == 1404
    assert comparison["production_is_globally_optimal_structure"] is True
    assert comparison["absolute_difference"] == 0.0


def test_reading_production_and_stage27_files_does_not_modify_them() -> None:
    production_path = REPO_ROOT / MINI_LOTO_PRODUCTION_PREDICTION
    stage27_path = REPO_ROOT / "data" / "prospective" / "stage27" / "MINI_LOTO" / "1404.json"

    before_production = _file_hash(production_path)
    before_mtime = production_path.stat().st_mtime_ns
    before_stage27 = _file_hash(stage27_path)
    before_stage27_mtime = stage27_path.stat().st_mtime_ns

    load_current_production_tickets(production_path)
    compare_production_to_optimum(production_path, MINI_LOTO)
    # Stage 27's record is read elsewhere in the project's own tooling, not
    # by this module at all -- Stage 30 never opens it. Assert that merely
    # running Stage 30's functions leaves it untouched regardless.
    json.loads(stage27_path.read_text(encoding="utf-8"))

    assert _file_hash(production_path) == before_production
    assert production_path.stat().st_mtime_ns == before_mtime
    assert _file_hash(stage27_path) == before_stage27
    assert stage27_path.stat().st_mtime_ns == before_stage27_mtime


def test_frozen_production_tickets_match_historical_snapshot() -> None:
    # Regression guard for the invariant Stage 30 actually depends on: the
    # #1404 ticket numbers were never regenerated or replaced. The record's
    # lifecycle status (PENDING -> EVALUATED) is expected to change on its
    # own as the live operational pipeline processes real draws over time;
    # that is normal external progress, not something Stage 30 controls or
    # should assume frozen. Only the ticket content itself is the frozen
    # snapshot this suite protects.
    production = load_current_production_tickets(REPO_ROOT / MINI_LOTO_PRODUCTION_PREDICTION)
    assert production["status"] in {"PENDING", "EVALUATED"}
    assert production["tickets"] == (
        (3, 11, 14, 19, 31),
        (2, 21, 22, 27, 30),
        (4, 5, 16, 23, 29),
    )


# ---------------------------------------------------------------------------
# G. Scenario 2 diagnostic: explicitly labeled, no lookahead
# ---------------------------------------------------------------------------


def test_model_weighted_diagnostic_is_labeled_and_uses_only_history_up_to_cutoff() -> None:
    diagnostic = model_weighted_diagnostic(
        history_path=REPO_ROOT / "data" / "processed" / "mini_loto_history.csv",
        production_path=REPO_ROOT / MINI_LOTO_PRODUCTION_PREDICTION,
        lottery=MINI_LOTO,
        n_simulations=500,
        seed=7,
    )
    assert diagnostic["label"] == "SCENARIO_2_MODEL_WEIGHTED_DIAGNOSTIC"
    assert "DIAGNOSTIC ONLY" in diagnostic["disclaimer"]
    assert "NOT evidence" in diagnostic["disclaimer"]
    assert diagnostic["cutoff_draw_number"] == 1403


def test_diagnostic_never_reads_beyond_declared_cutoff() -> None:
    from backend.app.research.data import load_draws_csv, validate_draw_sequence
    from backend.app.research.stage30_ticket_portfolio_optimization import (
        _recent_frequency_weights,
    )

    draws = load_draws_csv(REPO_ROOT / "data" / "processed" / "mini_loto_history.csv", MINI_LOTO)
    validate_draw_sequence(draws)
    cutoff = 1403
    weights_at_cutoff = _recent_frequency_weights(
        REPO_ROOT / "data" / "processed" / "mini_loto_history.csv", MINI_LOTO, cutoff, lookback=20
    )
    manual_window = [d for d in draws if d.draw_number <= cutoff][-20:]
    manual_counts = {n: 0 for n in range(1, 32)}
    for draw in manual_window:
        for number in draw.main_numbers:
            manual_counts[number] += 1
    assert weights_at_cutoff == manual_counts


# ---------------------------------------------------------------------------
# I. Ticket-count sensitivity
# ---------------------------------------------------------------------------


def test_ticket_count_sensitivity_disjoint_remains_best_1_through_6() -> None:
    sensitivity = ticket_count_sensitivity(6, MINI_LOTO)
    for n in range(2, 7):
        row = sensitivity[str(n)]
        assert row["disjoint_is_best_of_tested"] is True


def test_ticket_count_sensitivity_scales_disjoint_probability_up() -> None:
    sensitivity = ticket_count_sensitivity(6, MINI_LOTO)
    values = [sensitivity[str(n)]["disjoint"]["p_at_least_one_prize"]["float"] for n in range(1, 7)]
    assert values == sorted(values)  # strictly increasing with more tickets
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


# ---------------------------------------------------------------------------
# Region decomposition plumbing
# ---------------------------------------------------------------------------


def test_region_decomposition_covers_full_universe() -> None:
    owners, sizes = region_decomposition(disjoint_tickets(3), universe=31)
    assert sum(sizes) == 31
    assert sum(s for o, s in zip(owners, sizes, strict=True) if o) == 15


def test_evaluate_portfolio_exact_rejects_malformed_region_sizes() -> None:
    from backend.app.research.stage30_ticket_portfolio_optimization import evaluate_region_sizes

    with pytest.raises(ValueError):
        evaluate_region_sizes((frozenset({0}),), (5,), n_tickets=1, lottery=MINI_LOTO)
