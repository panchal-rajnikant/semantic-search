from src.retrieval.models import SearchResult
import numpy as np


class DenseRetriever:

    def __init__(
        self,
        embedding_service,
        vector_store
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        filters: dict | None = None
    ) -> list[SearchResult]:
        
        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )
        
        if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )
        
        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        results = []

        for embedding, metadata in zip(
            self.vector_store.embeddings,
            self.vector_store.metadata
        ):

            if not self._matches_filters(
                metadata,
                filters
            ):
                continue

            score = self._cosine_similarity(
                query_embedding,
                embedding
            )

            if score < similarity_threshold:
                continue
            
            results.append({
                **metadata,
                "score": score
            })

        results.sort(
            key=lambda item: item["score"],
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