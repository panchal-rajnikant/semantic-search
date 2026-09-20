from enum import Enum
from src.retrieval.models import SearchResult

class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RetrievalService:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        hybrid_retriever,
        reranker=None
    ):
        self.retrievers = {
            RetrievalStrategy.DENSE: dense_retriever,
            RetrievalStrategy.SPARSE: sparse_retriever,
            RetrievalStrategy.HYBRID: hybrid_retriever
        }

        self.reranker = reranker

    def search(
    self,
    query: str,
    strategy: RetrievalStrategy,
    top_k: int,
    candidate_k: int,
    similarity_threshold: float,
    filters: dict | None = None,
    rerank: bool = False
) -> list[SearchResult]:
        

        if not query.strip():
            raise ValueError("Query cannot be empty")

        retriever = self.retrievers[strategy]

        results = retriever.search(
            query=query,
            top_k=candidate_k,
            filters=filters,
            # threshold forwarded where applicable
        )

        if rerank and self.reranker:
            return self.reranker.rerank(
                query,
                results,
                top_k
            )

        results = results[:top_k]

        for rank, result in enumerate(results, 1):
            result.rank = rank

        return results