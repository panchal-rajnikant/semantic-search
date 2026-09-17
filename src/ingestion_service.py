from src.loader import load_documents
from src.chunker import chunk_documents
from src.config import SearchConfig

class IngestionService:

    def __init__(self, EmbeddingService, vector_store):
        self.embedding_service = EmbeddingService
        self.vector_store = vector_store
        self.config = SearchConfig()

    def ingest(self, data_dir: dir) -> None:
        # load
        documents = load_documents(data_dir)
        # chunk
        chunks = chunk_documents(documents, self.config.chunk_size, self.config.chunk_overlap) 
        texts = [chunk["text"] for chunk in chunks]
        
        # embedding
        embeddings =(self.embedding_service.embed_texts(texts))

        # store
        self.vector_store.save(
            embeddings=embeddings,
            metadata=chunks
        )