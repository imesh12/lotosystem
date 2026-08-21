from dataclasses import asdict, dataclass

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.backtesting import BacktestResult, run_backtest
from backend.app.research.candidates import Candidate, generate_candidates
from backend.app.research.config import CandidateStrategy, ResearchConfig
from backend.app.research.data import HistoricalDraw
from backend.app.research.dataset import calculate_dataset_hash, validate_lottery_dataset
from backend.app.research.statistics import StatisticsBundle, calculate_statistics


@dataclass(frozen=True, slots=True)
class ProvenanceSummary:
    sources: tuple[str, ...]
    source_urls: tuple[str, ...]
    source_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchResult:
    lottery: str
    dataset_version: str
    dataset_hash: str
    strategy: CandidateStrategy
    configuration: dict[str, object]
    statistics: StatisticsBundle
    candidates: tuple[Candidate, ...]
    backtest: BacktestResult
    provenance: ProvenanceSummary
    warnings: tuple[str, ...]


def run_research(
    draws: tuple[HistoricalDraw, ...] | list[HistoricalDraw],
    lottery: LotteryDefinition,
    config: ResearchConfig,
    strategy: CandidateStrategy,
) -> ResearchResult:
    ordered = validate_lottery_dataset(draws, lottery)
    statistics = calculate_statistics(ordered, lottery, config)
    candidates = generate_candidates(lottery, statistics, config, strategy)
    backtest = run_backtest(ordered, lottery, config, strategy)
    return ResearchResult(
        lottery=str(lottery.code),
        dataset_version=config.dataset_version,
        dataset_hash=calculate_dataset_hash(ordered),
        strategy=strategy,
        configuration=asdict(config),
        statistics=statistics,
        candidates=candidates,
        backtest=backtest,
        provenance=_provenance_summary(ordered),
        warnings=(),
    )


def _provenance_summary(draws: tuple[HistoricalDraw, ...]) -> ProvenanceSummary:
    return ProvenanceSummary(
        sources=tuple(sorted({draw.source for draw in draws if draw.source})),
        source_urls=tuple(sorted({draw.source_url for draw in draws if draw.source_url})),
        source_hashes=tuple(sorted({draw.content_hash for draw in draws if draw.content_hash})),
    )
