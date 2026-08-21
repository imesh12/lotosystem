from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.baselines import RandomBaselineSummary, run_random_baseline_replications
from backend.app.research.candidates import Candidate, generate_candidates
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import validate_lottery_dataset
from backend.app.research.metrics import BacktestMetrics, calculate_match_metrics
from backend.app.research.prize import PrizeMatchResult, match_ticket
from backend.app.research.statistics import calculate_statistics


@dataclass(frozen=True, slots=True)
class TicketEvaluation:
    candidate: Candidate
    match_result: PrizeMatchResult

    @property
    def match_count(self) -> int:
        return self.match_result.main_match_count


@dataclass(frozen=True, slots=True)
class BacktestStep:
    target_draw_number: int
    target_draw_date: str
    strategy_evaluations: tuple[TicketEvaluation, ...]
    fixed_baseline_evaluations: tuple[TicketEvaluation, ...]
    training_draw_count: int
    lookahead_safe: bool

    @property
    def strategy_candidate(self) -> Candidate:
        return self.strategy_evaluations[0].candidate

    @property
    def strategy_matches(self) -> int:
        return self.strategy_evaluations[0].match_count

    @property
    def baseline_candidate(self) -> Candidate:
        return self.fixed_baseline_evaluations[0].candidate

    @property
    def baseline_matches(self) -> int:
        return self.fixed_baseline_evaluations[0].match_count


@dataclass(frozen=True, slots=True)
class StrategyRandomComparison:
    strategy_average_matches: float
    random_average_matches: float
    average_match_difference: float
    strategy_prize_qualified_rate: float
    random_prize_qualified_rate: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy: CandidateStrategy
    baseline_strategy: CandidateStrategy
    steps: tuple[BacktestStep, ...]
    strategy_metrics: BacktestMetrics
    baseline_metrics: BacktestMetrics
    random_baseline: RandomBaselineSummary
    comparison: StrategyRandomComparison
    lookahead_safe: bool


def run_backtest(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    strategy: CandidateStrategy,
) -> BacktestResult:
    ordered = validate_lottery_dataset(draws, lottery)
    steps: list[BacktestStep] = []

    for target_index in range(config.backtest_min_training_draws, len(ordered)):
        training_draws = ordered[:target_index]
        target_draw = ordered[target_index]
        if config.evaluation_start and target_draw.draw_date < config.evaluation_start:
            continue
        if config.evaluation_end and target_draw.draw_date > config.evaluation_end:
            continue
        lookahead_safe = all(
            (training_draw.draw_date, training_draw.draw_number)
            < (target_draw.draw_date, target_draw.draw_number)
            for training_draw in training_draws
        )
        stats = calculate_statistics(training_draws, lottery, config)
        strategy_candidates = generate_candidates(
            lottery,
            stats,
            config,
            strategy,
            limit=config.backtest_candidate_count,
        )
        baseline_candidates = generate_candidates(
            lottery,
            stats,
            config,
            CandidateStrategy.FIXED_BASELINE,
            limit=1,
        )
        if not strategy_candidates:
            continue
        steps.append(
            BacktestStep(
                target_draw_number=target_draw.draw_number,
                target_draw_date=target_draw.draw_date.isoformat(),
                strategy_evaluations=tuple(
                    _evaluate_candidate(candidate, target_draw, lottery)
                    for candidate in strategy_candidates[: config.backtest_candidate_count]
                ),
                fixed_baseline_evaluations=tuple(
                    _evaluate_candidate(candidate, target_draw, lottery)
                    for candidate in baseline_candidates
                ),
                training_draw_count=len(training_draws),
                lookahead_safe=lookahead_safe,
            )
        )

    target_draws = tuple(
        draw
        for draw in ordered[config.backtest_min_training_draws :]
        if (not config.evaluation_start or draw.draw_date >= config.evaluation_start)
        and (not config.evaluation_end or draw.draw_date <= config.evaluation_end)
    )
    strategy_matches = tuple(
        evaluation.match_count for step in steps for evaluation in step.strategy_evaluations
    )
    strategy_prizes = tuple(
        evaluation.match_result.qualifies_for_prize
        for step in steps
        for evaluation in step.strategy_evaluations
    )
    baseline_matches = tuple(
        evaluation.match_count for step in steps for evaluation in step.fixed_baseline_evaluations
    )
    baseline_prizes = tuple(
        evaluation.match_result.qualifies_for_prize
        for step in steps
        for evaluation in step.fixed_baseline_evaluations
    )
    strategy_metrics = calculate_match_metrics(
        strategy_matches, lottery.numbers_per_ticket, strategy_prizes
    )
    random_baseline = run_random_baseline_replications(target_draws, lottery, config)
    return BacktestResult(
        strategy=strategy,
        baseline_strategy=CandidateStrategy.FIXED_BASELINE,
        steps=tuple(steps),
        strategy_metrics=strategy_metrics,
        baseline_metrics=calculate_match_metrics(
            baseline_matches, lottery.numbers_per_ticket, baseline_prizes
        ),
        random_baseline=random_baseline,
        comparison=StrategyRandomComparison(
            strategy_average_matches=strategy_metrics.average_matches,
            random_average_matches=random_baseline.mean_matches,
            average_match_difference=(
                strategy_metrics.average_matches - random_baseline.mean_matches
            ),
            strategy_prize_qualified_rate=strategy_metrics.prize_qualified_rate,
            random_prize_qualified_rate=random_baseline.prize_qualified_rate,
        ),
        lookahead_safe=all(step.lookahead_safe for step in steps),
    )


def _evaluate_candidate(
    candidate: Candidate,
    target_draw: HistoricalDraw,
    lottery: LotteryDefinition,
) -> TicketEvaluation:
    return TicketEvaluation(
        candidate=candidate,
        match_result=match_ticket(candidate.numbers, target_draw, lottery),
    )
