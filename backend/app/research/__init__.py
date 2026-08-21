"""Deterministic statistical research engine."""

from backend.app.research.backtesting import BacktestResult, run_backtest
from backend.app.research.baselines import (
    RandomBaselineSummary,
    generate_uniform_random_ticket,
    run_random_baseline_replications,
)
from backend.app.research.candidates import (
    Candidate,
    CandidateScore,
    generate_candidates,
    score_candidate,
)
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import HistoricalDraw, load_draws_csv, validate_draw_sequence
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.features import CandidateFeatures, NumberFeature, build_candidate_features
from backend.app.research.pipeline import ResearchResult, run_research
from backend.app.research.prize import PrizeMatchResult, match_ticket
from backend.app.research.statistics import StatisticsBundle, calculate_statistics

__all__ = [
    "BacktestResult",
    "Candidate",
    "CandidateFeatures",
    "CandidateScore",
    "CandidateStrategy",
    "HistoricalDraw",
    "NumberFeature",
    "ResearchConfig",
    "ResearchResult",
    "PrizeMatchResult",
    "RandomBaselineSummary",
    "StatisticsBundle",
    "build_candidate_features",
    "calculate_statistics",
    "calculate_dataset_hash",
    "generate_candidates",
    "generate_uniform_random_ticket",
    "load_draws_csv",
    "run_backtest",
    "run_research",
    "run_random_baseline_replications",
    "score_candidate",
    "match_ticket",
    "validate_lottery_dataset",
    "validate_draw_sequence",
]
