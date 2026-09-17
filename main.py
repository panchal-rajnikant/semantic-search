from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.search_service import SearchService
from src.config import SearchConfig

def main():
    config = SearchConfig()
    embedding_service = EmbeddingService(config.embedding_model)
    vector_store = VectorStore(config.storage_dir)

    vector_store.load()

    search_service = SearchService(
        embedding_service,
        vector_store
    )

    query = input(
        "Enter search query: "
    )

    results = search_service.search(
        query = query,
        top_k = config.top_k,
        similarity_threshold = config.similarity_threshold
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