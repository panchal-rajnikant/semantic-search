from sentence_transformers import CrossEncoder

from src.retrieval.models import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str, top_k: int = 3):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
        if not candidates:
            return []

        candidate_pairs = [
            (query, candidate.text)
            for candidate in candidates
        ]

        scores = self.model.predict(candidate_pairs)

        ranked = []
        for candidate, score in zip(candidates, scores):
            ranked_candidate = dict(candidate.__dict__)
            ranked_candidate["score"] = float(score)
            ranked.append(ranked_candidate)

        ranked.sort(key=lambda item: item["score"], reverse=True)

        limit = top_k if top_k is not None else self.top_k
        return ranked[:limit]
