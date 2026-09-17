from src.loader import load_documents
from src.chunker import chunk_documents

class IngestionService:

    def __init__(self, EmbeddingService, vector_store, config):
        self.embedding_service = EmbeddingService
        self.vector_store = vector_store
        self.config = config

    def ingest(self, data_dir: dir) -> None:
        # load
        documents = load_documents(data_dir)
        # chunk
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.overlap) 
        texts = [chunk["text"] for chunk in chunks]
        
        # embedding
        embeddings =(self.embedding_service.embed_texts(texts))

        # store
        self.vector_store.save(
            embeddings=embeddings,
            metadata=chunks
        )