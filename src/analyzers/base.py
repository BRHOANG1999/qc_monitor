"""Abstract base class for analyzers."""

from abc import ABC, abstractmethod
from src.utils.mat_loader import ChunkData


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, chunk: ChunkData, file_id: int) -> dict:
        """Run analysis on a chunk. Returns dict of metrics."""
        ...
