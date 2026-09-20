class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        dense_weight: float = 0.5
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight

    def search(self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        filters: dict | None = None):

        dense_results = self.dense.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters
        )

        sparse_results = self.sparse.search(
            query=query,
            top_k=top_k
        )

        dense_scores = [
            result["score"]
            for result in dense_results
        ]

        sparse_scores = [
            result["score"]
            for result in sparse_results
        ]

        normalized_dense_scores = self.normalize(
            dense_scores
        )

        normalized_sparse_scores = self.normalize(
            sparse_scores
        )

        merged = {}
        # add dense result 
        for result, normalized_score in zip(
            dense_results,
            normalized_dense_scores
        ):
            key = (
                result["source"],
                result["chunk_id"]
            )

            merged[key] = {
                "result": result,
                "dense_score": normalized_score,
                "sparse_score": 0.0
            }

        # add sparse result
        for result, normalized_score in zip(
                sparse_results,
                normalized_sparse_scores
            ):
                key = (
                        result["source"],
                    result["chunk_id"]
                )

                if key in merged:
                    merged[key]["sparse_score"] = normalized_score
                else:
                    merged[key] = {
                        "result": result,
                        "dense_score": 0.0,
                        "sparse_score": normalized_score
                    }
                            
        results = []
        
        for item in merged.values():

            dense_score = item["dense_score"]
            sparse_score = item["sparse_score"]

            hybrid_score = (
                self.dense_weight * dense_score
                + (1 - self.dense_weight) * sparse_score
            )

            item["hybrid_score"] = hybrid_score

        ranked = sorted(
            merged.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True
        )
        ranked = ranked[:top_k]

        results = []
        for item in ranked:
            result = {
                **item["result"],
                "score": item["hybrid_score"]
            }
            results.append(result)

        return results

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    
        