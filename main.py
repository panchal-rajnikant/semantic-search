from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.search_service import SearchService


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        "storage"
    )

    vector_store.load()

    search_service = SearchService(
        embedding_service,
        vector_store
    )

    query = input(
        "Enter search query: "
    )

    results = search_service.search(
        query=query,
        top_k=3
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