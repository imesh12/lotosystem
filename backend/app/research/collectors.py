from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.app.domain.lottery import LotteryDefinition
from backend.app.research.data import HistoricalDraw, load_draws_csv


class CollectorInterface(ABC):
    @abstractmethod
    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        """Collect historical draws for a lottery."""


class CSVCollector(CollectorInterface):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def collect(self, lottery: LotteryDefinition) -> tuple[HistoricalDraw, ...]:
        return load_draws_csv(self.path, lottery)
