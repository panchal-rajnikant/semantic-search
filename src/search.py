import numpy as np

class searchService:

    def __init__(self, embedding_service, vector_index):
        self.embedding_service = embedding_service
        self.vector_index = vector_index

    def search(self, query: str, top_k:int = 3, filters: dict | None = None):

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_embedding = self.embedding_service.embed_query(query)

        return self.vector_index.search( query_embedding=query_embedding, top_k=top_k, filters=filters)
