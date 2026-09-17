from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.ingestion_service import IngestionService
from src.config import SearchConfig

def main():
    config = SearchConfig()
    embedding_service = EmbeddingService(config.embedding_model)

    vector_store = VectorStore(config.storage_dir)

    #initialize ingestion service 
    ingestion_service = IngestionService(
        embedding_service,
        vector_store
    )

    # load-chunk-text-embed-save 
    ingestion_service.ingest(config.data_dir)

    print(
        "Vector index created successfully."
    )


if __name__ == "__main__":
    main()