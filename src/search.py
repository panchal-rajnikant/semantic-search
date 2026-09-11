import numpy as np


def cosine_similarity(
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


class SemanticSearch:

    def __init__(
        self,
        chunks: list[dict],
        embeddings,
        embedding_service
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Each chunk must have an embedding"
            )

        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_service = embedding_service


    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict]:

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

        for chunk, embedding in zip(
            self.chunks,
            self.embeddings
        ):

            score = cosine_similarity(
                query_embedding,
                embedding
            )

            results.append(
                {
                    **chunk,
                    "score": score
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]