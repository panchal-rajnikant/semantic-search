import numpy as np

class vectorIndex:

    def __init__(self):
        self.items = []

    def add(self,embedding, metadata):
        self.items.append({
            "embedding": embedding,
            "metadata": metadata
        })


    def search(self, query_embedding, top_k: int = 3) ->list[dict]:
        results = []
        for item in self.items:
            score = self.cosine_similarity(query_embedding, item["embedding"])

            results.append({
                **item["metadata"],
                "score": score
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        
        return results[:top_k]


    @staticmethod
    def cosine_similarity(vector_a, vector_b)-> float:
        denominator = ( np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

        if denominator == 0:
            return 0.0
        
        return float(np.dot(vector_a, vector_b) / denominator)