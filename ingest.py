from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.ingestion_service import IngestionService


def main():

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        "storage"
    )

    #initialize ingestion service 
    ingestion_service = IngestionService(
        embedding_service,
        vector_store
    )

    # load-chunk-text-embed-save 
    ingestion_service.ingest(
        "data"
    )

    print(
        "Vector index created successfully."
    )


if __name__ == "__main__":
    main()