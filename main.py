from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.service import RetrievalService
from src.retrieval.hybrid import HybridRetriever

from src.config import SearchConfig, RetrievalConfig


def main():
    search_config = SearchConfig()
    retrieval_config = RetrievalConfig()

    embedding_service = EmbeddingService(search_config.embedding_model)
    vector_store = VectorStore(search_config.storage_dir)

    vector_store.load()
    chunks = vector_store.metadata

    # Retrievers
    dense_retriever  = DenseRetriever(
        embedding_service,
        vector_store
    )

    sparse_retriever = SparseRetriever(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=retrieval_config.dense_weight
    )

    # Application service
    retrieval_service = RetrievalService(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        hybrid_retriever=hybrid_retriever
    )

    query = input(
        "Enter search query: "
    )
    results = retrieval_service.search(
        query=query,
        strategy=retrieval_config.strategy,
        top_k=search_config.top_k,
        candidate_k = retrieval_config.candidate_k,
        similarity_threshold=search_config.similarity_threshold
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