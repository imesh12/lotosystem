from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw
from backend.app.research.exceptions import ResearchValidationError
from backend.app.research.extra_trees_evaluation import benjamini_hochberg_adjust_p_values
from backend.app.research.persistence import research_result_json
from backend.app.research.settlement import DrawSettlement
from backend.app.research.stage28_ticket_popularity import (
    MINI_LOTO_COMBINATION_COUNT,
    RECOMMENDATION_NONE,
    WinnerCountObservation,
    component_association_tests,
    conditional_payout_examples,
    enumerate_mini_loto_combinations,
    holm_adjust_p_values,
    popularity_components,
    primary_association_test,
    proxy_feature_definitions,
    recommendation_from_association,
    run_stage28_ticket_popularity_research,
    score_historical_winners,
    score_ticket,
    score_universe,
    spearman_correlation,
    universe_summary,
)


def _draws(
    lottery: LotteryDefinition = MINI_LOTO,
    *,
    start_number: int = 543,
    count: int = 12,
    start_date: date = date(2010, 1, 5),
) -> tuple[HistoricalDraw, ...]:
    rows: list[HistoricalDraw] = []
    for index in range(count):
        base = ((index * 7) % lottery.number_max) + 1
        main = tuple(
            sorted(
                ((base + offset * 5 - 1) % lottery.number_max) + 1
                for offset in range(lottery.numbers_per_ticket)
            )
        )
        bonus = next(
            number
            for number in range(lottery.number_min, lottery.number_max + 1)
            if number not in main
        )
        rows.append(
            HistoricalDraw(
                lottery=lottery,
                draw_number=start_number + index,
                draw_date=start_date + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(rows)


def _settlement_payload(draw: HistoricalDraw, *, winners: int, payout: int) -> DrawSettlement:
    return {
        "schema_version": "stage12-paper-settlement-v1",
        "lottery": str(draw.lottery.code),
        "draw_number": draw.draw_number,
        "draw_date": draw.draw_date.isoformat(),
        "prediction_record_path": "prediction.json",
        "prediction_generated_at": "2026-01-01T00:00:00+00:00",
        "prediction_dataset_hash": "hash",
        "settled_at": "2026-01-02T00:00:00+00:00",
        "actual_main_numbers": draw.main_numbers,
        "actual_bonus_numbers": draw.bonus_numbers,
        "payouts": [
            {
                "lottery": str(draw.lottery.code),
                "draw_number": draw.draw_number,
                "prize_tier": "1st",
                "payout_yen": payout,
                "winners_count": winners,
                "source": "test",
                "source_url": None,
                "retrieved_at": None,
            }
        ],
        "tickets": [],
        "ticket_count": 0,
        "ticket_price_yen": draw.lottery.ticket_price_yen,
        "paper_total_cost_yen": 0,
        "paper_gross_winnings_yen": 0,
        "paper_net_yen": 0,
        "financial_status": "COMPLETE",
        "warnings": [],
    }


def _write_settlement(root: Path, draw: HistoricalDraw, *, winners: int, payout: int) -> None:
    path = root / "MINI_LOTO" / f"{draw.draw_number}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        research_result_json(_settlement_payload(draw, winners=winners, payout=payout)),
        encoding="utf-8",
    )


def test_stage28_is_mini_loto_only(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError, match="MINI_LOTO only"):
        run_stage28_ticket_popularity_research(
            _draws(LOTO6, start_number=2000),
            LOTO6,
            output_dir=None,
            settlement_root=tmp_path / "settlements",
        )


def test_enumerates_exact_mini_loto_universe_and_valid_tickets() -> None:
    combinations = enumerate_mini_loto_combinations()

    assert len(combinations) == MINI_LOTO_COMBINATION_COUNT
    assert combinations[0] == (1, 2, 3, 4, 5)
    assert combinations[-1] == (27, 28, 29, 30, 31)
    assert all(tuple(sorted(ticket)) == ticket for ticket in combinations)
    assert all(len(set(ticket)) == 5 for ticket in combinations)
    assert all(1 <= number <= 31 for ticket in combinations for number in ticket)


def test_score_and_components_are_deterministic() -> None:
    first = score_ticket((3, 11, 18, 24, 29))
    second = score_ticket((3, 11, 18, 24, 29))

    assert first == second
    assert set(first.components) == {definition.name for definition in proxy_feature_definitions()}
    assert all(0 <= value <= 1 for value in first.components.values())


def test_obvious_sequence_scores_more_patterned_than_irregular_control() -> None:
    sequence = score_ticket((1, 2, 3, 4, 5))
    irregular = score_ticket((3, 11, 18, 24, 29))

    assert sequence.normalized_score > irregular.normalized_score
    assert sequence.components["simple_sequence_indicator"] == 1.0
    assert irregular.components["simple_sequence_indicator"] == 0.0


def test_ticket_order_permutation_cannot_change_score() -> None:
    assert score_ticket((29, 3, 24, 11, 18)) == score_ticket((3, 11, 18, 24, 29))


def test_historical_result_does_not_affect_proxy_definition() -> None:
    before = popularity_components((3, 11, 18, 24, 29))
    _ = score_historical_winners(_draws(), score_universe())
    after = popularity_components((3, 11, 18, 24, 29))

    assert before == after


def test_winner_count_observations_use_sales_normalization_when_available() -> None:
    observations = (
        WinnerCountObservation(
            draw_number=1,
            draw_date="2026-01-01",
            main_numbers=(1, 2, 3, 4, 5),
            popularity_score=0.9,
            first_prize_winners=9,
            first_prize_payout_yen=1,
            sales_amount_yen=1_800,
            estimated_tickets_sold=9,
            normalized_winner_rate=1.0,
        ),
        WinnerCountObservation(
            draw_number=2,
            draw_date="2026-01-08",
            main_numbers=(3, 11, 18, 24, 29),
            popularity_score=0.1,
            first_prize_winners=1,
            first_prize_payout_yen=1,
            sales_amount_yen=2_000,
            estimated_tickets_sold=10,
            normalized_winner_rate=0.1,
        ),
    )

    assert observations[0].normalized_winner_rate == 1.0
    assert observations[1].normalized_winner_rate == 0.1


def test_primary_association_requires_sales_and_enough_observations(tmp_path: Path) -> None:
    result = run_stage28_ticket_popularity_research(
        _draws(),
        MINI_LOTO,
        output_dir=None,
        settlement_root=tmp_path / "settlements",
    )

    assert result.primary_association.classification == "INCONCLUSIVE"
    assert result.primary_association.raw_p_value == 1.0
    assert result.recommendation == RECOMMENDATION_NONE


def test_association_analysis_is_deterministic_with_sales_normalized_data() -> None:
    observations = tuple(
        WinnerCountObservation(
            draw_number=index,
            draw_date=f"2026-01-{index:02d}",
            main_numbers=(1, 2, 3, 4, 5),
            popularity_score=float(index),
            first_prize_winners=index,
            first_prize_payout_yen=1,
            sales_amount_yen=20_000,
            estimated_tickets_sold=100,
            normalized_winner_rate=index / 100,
        )
        for index in range(1, 7)
    )

    first = primary_association_test(observations, seed=123)
    second = primary_association_test(observations, seed=123)

    assert first == second
    assert first.usable_observations == 6
    assert first.effect == pytest.approx(1.0)


def test_no_future_leakage_in_historical_association_analysis(tmp_path: Path) -> None:
    draws = _draws(count=3)
    _write_settlement(tmp_path / "settlements", draws[0], winners=1, payout=1)
    first = run_stage28_ticket_popularity_research(
        draws[:1],
        MINI_LOTO,
        output_dir=None,
        settlement_root=tmp_path / "settlements",
    )
    second = run_stage28_ticket_popularity_research(
        draws,
        MINI_LOTO,
        output_dir=None,
        settlement_root=tmp_path / "settlements",
    )

    assert first.winner_count_observations == second.winner_count_observations


def test_stage27_production_and_settlement_files_are_untouched(tmp_path: Path) -> None:
    draws = _draws()
    protected_roots = (tmp_path / "prospective", tmp_path / "predictions", tmp_path / "settlements")
    for root in protected_roots:
        root.mkdir(parents=True)
        (root / "sentinel.json").write_text('{"unchanged": true}', encoding="utf-8")
    before = {
        root: sorted(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
        for root in protected_roots
    }

    run_stage28_ticket_popularity_research(
        draws,
        MINI_LOTO,
        output_dir=tmp_path / "exports",
        settlement_root=tmp_path / "empty_settlements",
    )

    after = {
        root: sorted(path.read_text(encoding="utf-8") for path in root.glob("*.json"))
        for root in protected_roots
    }
    assert after == before


def test_holm_and_bh_adjustments_are_deterministic() -> None:
    raw = {"a": 0.02, "b": 0.04, "c": 0.20}

    assert holm_adjust_p_values(raw) == holm_adjust_p_values(raw)
    assert benjamini_hochberg_adjust_p_values(raw) == benjamini_hochberg_adjust_p_values(raw)


def test_recommendation_gate_is_deterministic_for_inconclusive_result(tmp_path: Path) -> None:
    result = run_stage28_ticket_popularity_research(
        _draws(),
        MINI_LOTO,
        output_dir=None,
        settlement_root=tmp_path / "settlements",
    )

    assert recommendation_from_association(result.primary_association) == RECOMMENDATION_NONE
    assert result.anti_popularity_selector_built is False


def test_universe_summary_is_compact_and_reproducible() -> None:
    scores = score_universe()
    first = universe_summary(scores)
    second = universe_summary(scores)

    assert first == second
    assert first.combination_count == MINI_LOTO_COMBINATION_COUNT
    assert len(first.highest_risk_examples) == 10
    assert len(first.lowest_risk_examples) == 10


def test_conditional_payout_examples_do_not_change_hit_probability() -> None:
    examples = conditional_payout_examples(prize_pool_yen=10_000_000, split_counts=(1, 2, 10))

    assert examples["exact_five_number_hit_probability"] == "1/169911"
    assert examples["examples"]["1"]["conditional_payout_per_winning_ticket_yen"] == 10_000_000
    assert examples["examples"]["10"]["conditional_payout_per_winning_ticket_yen"] == 1_000_000


def test_output_payload_is_json_serializable_and_compact(tmp_path: Path) -> None:
    result = run_stage28_ticket_popularity_research(
        _draws(),
        MINI_LOTO,
        output_dir=tmp_path / "stage28",
        settlement_root=tmp_path / "settlements",
    )
    payload = json.loads(
        (tmp_path / "stage28" / "mini_loto_ticket_popularity_report.json").read_text()
    )

    assert payload["universe_summary"]["combination_count"] == MINI_LOTO_COMBINATION_COUNT
    assert payload["recommendation"] == result.recommendation
    assert (tmp_path / "stage28" / "mini_loto_popularity_universe_compact.json").exists()


def test_spearman_handles_equal_values() -> None:
    assert spearman_correlation((1.0, 1.0, 1.0), (1.0, 2.0, 3.0)) == 0.0


def test_component_associations_are_inconclusive_without_sales() -> None:
    associations = component_association_tests((), seed=123)

    assert associations
    assert all(result.classification == "INCONCLUSIVE" for result in associations.values())
