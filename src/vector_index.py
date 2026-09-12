import numpy as np

class vectorIndex:

    def __init__(self):
        self.items = []

    def add(self,embedding, metadata):
        self.items.append({
            "embedding": embedding,
            "metadata": metadata
        })


    def search(self, query_embedding, top_k: int = 3, filters: dict | None = None) ->list[dict]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )
        
        results = []

        for item in self.items:
            metadata = item["metadata"]

            if not self._matches_filters(metadata, filters):
                continue

            score = self.cosine_similarity(query_embedding, item["embedding"])

            results.append({
                **metadata,
                "score": score
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        
        return results[:top_k]

    @staticmethod
    def _matches_filters(metadata: dict, filters:dict | None):

        if not filters:
            return True

        for key, expected_value in filters.items():

            if metadata.get(key) != expected_value:
                return False

        return True

        
    @staticmethod
    def cosine_similarity(vector_a, vector_b)-> float:
        denominator = ( np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

        if denominator == 0:
            return 0.0
        
        return float(np.dot(vector_a, vector_b) / denominator)