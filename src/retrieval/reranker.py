from src.retrieval.models import SearchResult
from abc import ABC, abstractmethod


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]: