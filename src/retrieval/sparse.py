import numpy as np
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

@dataclass
class SearchResult:
    text: str
    source: str
    document_id: str
    chunk_id: int
    score: float

class SparseRetriever:

    def __init__(self, chunks: list[dict] ):
        self.chunks = chunks

        tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int,
        strategy: str = "dense",
    ) -> list[SearchResult]:

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )
        
        if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )

        query_tokens = self._tokenize(query)
        
        scores = self.index.get_scores(
            query_tokens
        )
        
        results = []

        for chunk, score in zip(self.chunks, scores):

            results.append({
                **chunk,
                "bm25_score": float(score)
            })

        results.sort(
            key=lambda item: item["bm25_score"],
            reverse=True
        )

        return results[:top_k]

    @staticmethod
    def _matches_filters(
        metadata: dict,
        filters: dict | None
    ) -> bool:

        if not filters:
            return True

        for key, expected in filters.items():
            if metadata.get(key) != expected:
                return False

        return True

    @staticmethod
    def _cosine_similarity(
        vector_a,
        vector_b
    ) -> float:

        denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(vector_a, vector_b)
            / denominator
        )