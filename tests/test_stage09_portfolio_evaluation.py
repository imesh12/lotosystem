from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from backend.app.domain import LOTO6, MINI_LOTO
from backend.app.research.config import ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.portfolio_evaluation import (
    build_candidate_pool,
    construct_portfolio,
    overlap_count,
    run_stage09_portfolio_evaluation,
    save_stage09_portfolio_evaluation,
    unique_coverage,
)


def _loto6_draws(count: int = 14) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 20) + 1
        main = tuple(sorted(((start + offset * 5 - 1) % 43) + 1 for offset in range(6)))
        bonus = next(number for number in range(1, 44) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=LOTO6,
                draw_number=index + 1,
                draw_date=date(2026, 1, 1) + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _mini_draws(count: int = 14) -> tuple[HistoricalDraw, ...]:
    draws: list[HistoricalDraw] = []
    for index in range(count):
        start = (index % 16) + 1
        main = tuple(sorted(((start + offset * 4 - 1) % 31) + 1 for offset in range(5)))
        bonus = next(number for number in range(1, 32) if number not in main)
        draws.append(
            HistoricalDraw(
                lottery=MINI_LOTO,
                draw_number=index + 1,
                draw_date=date(2026, 1, 1) + timedelta(days=index * 7),
                main_numbers=main,
                bonus_numbers=(bonus,),
            )
        )
    return tuple(draws)


def _scores(lottery_max: int) -> dict[int, float]:
    return {number: float(lottery_max + 1 - number) for number in range(1, lottery_max + 1)}


def _config() -> ResearchConfig:
    return ResearchConfig(seed=123456, backtest_min_training_draws=3)


def _write_draws_csv(path: Path, draws: tuple[HistoricalDraw, ...]) -> None:
    rows = ["lottery,draw_number,draw_date,main_numbers,bonus_numbers"]
    for draw in draws:
        rows.append(
            ",".join(
                (
                    str(draw.lottery.code),
                    str(draw.draw_number),
                    draw.draw_date.isoformat(),
                    " ".join(str(number) for number in draw.main_numbers),
                    " ".join(str(number) for number in draw.bonus_numbers),
                )
            )
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_overlap_and_coverage_helpers() -> None:
    tickets = ((1, 2, 3, 4, 5, 6), (4, 5, 6, 7, 8, 9))

    assert overlap_count(*tickets) == 3
    assert unique_coverage(tickets) == 9


def test_candidate_pool_and_top_ranked_portfolio_are_deterministic() -> None:
    scores = _scores(43)

    first_pool = build_candidate_pool(scores, LOTO6, candidate_pool_size=10)
    second_pool = build_candidate_pool(scores, LOTO6, candidate_pool_size=10)
    portfolio = construct_portfolio(scores, LOTO6, "top_ranked")

    assert first_pool == second_pool
    assert len(first_pool) == 10
    assert portfolio.tickets == ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12))
    assert portfolio.overlap_count == 0
    assert portfolio.unique_number_coverage == 12


def test_overlap_penalty_and_coverage_methods_produce_valid_distinct_tickets() -> None:
    scores = _scores(43)

    diversified = construct_portfolio(scores, LOTO6, "diversified")
    coverage = construct_portfolio(scores, LOTO6, "coverage")
    penalty = construct_portfolio(scores, LOTO6, "overlap_penalty_1")

    for portfolio in (diversified, coverage, penalty):
        assert len(portfolio.tickets) == 2
        assert portfolio.tickets[0] != portfolio.tickets[1]
        assert all(len(ticket) == LOTO6.numbers_per_ticket for ticket in portfolio.tickets)
        assert portfolio.unique_number_coverage >= LOTO6.numbers_per_ticket


def test_stage09_result_is_reproducible_and_leakage_safe(tmp_path: Path) -> None:
    first = run_stage09_portfolio_evaluation(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
        candidate_pool_size=10,
    )
    second = run_stage09_portfolio_evaluation(
        _loto6_draws(),
        LOTO6,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
        candidate_pool_size=10,
    )
    output = tmp_path / "stage09.json"
    save_stage09_portfolio_evaluation(first, output)

    assert first == second
    assert output.exists()
    assert first.leakage.lookahead_safe is True
    assert "top_ranked" in first.method_results
    assert "coverage" in first.method_results


def test_stage09_supports_mini_loto() -> None:
    result = run_stage09_portfolio_evaluation(
        _mini_draws(),
        MINI_LOTO,
        _config(),
        bootstrap_replications=20,
        ml_min_training_draws=8,
        ml_refit_interval=3,
        candidate_pool_size=10,
    )

    assert result.lottery == "MINI_LOTO"
    assert result.feature_group == "pair_only"
    assert result.leakage.lookahead_safe is True


def test_stage09_cli_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini.csv"
    _write_draws_csv(csv_path, _mini_draws())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.research.cli",
            "--lottery",
            "MINI_LOTO",
            "--data",
            str(csv_path),
            "--seed",
            "123456",
            "--bootstrap-replications",
            "20",
            "--ml-min-training-draws",
            "8",
            "--ml-refit-interval",
            "3",
            "portfolio-evaluation",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["leakage"]["lookahead_safe"] is True
    assert payload["portfolio_version"] == "two-ticket-portfolio-v1"
