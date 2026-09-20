from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.service import RetrievalService
# from src.retrieval.hybrid import HybridRetriever

from src.config import SearchConfig

def main():
    config = SearchConfig()
    embedding_service = EmbeddingService(config.embedding_model)
    vector_store = VectorStore(config.storage_dir)

    vector_store.load()

    # Retrievers
    dense_retriever  = DenseRetriever(
        embedding_service,
        vector_store
    )

    sparse_retriever = SparseRetriever(
        vector_store.metadata
    )

    # hybrid_retriever = HybridRetriever(
    #     dense_retriever,
    #     sparse_retriever
    # )

    # Application service
    retrieval_service = RetrievalService(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        # hybrid_retriever=hybrid_retriever
    )

    query = input(
        "Enter search query: "
    )
    results = retrieval_service.search(
        query=query,
        strategy=config.strategy,
        top_k=config.top_k,
        similarity_threshold=config.similarity_threshold
        )

    for rank, result in enumerate(
        results,
        start=1
    ):
        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            result["text"]
        )


if __name__ == "__main__":
    main()