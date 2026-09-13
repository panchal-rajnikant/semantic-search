from src.loader import load_documents
from src.chunker import chunk_documents

class IngestionService:

    def __init__(self, EmbeddingService, vector_store):
        self.embedding_service = EmbeddingService
        self.vector_store = vector_store

    def ingest(self, data_dir: dir) -> None:
        # load
        documents = load_documents(data_dir)
        # chunk
        chunks = chunk_documents(documents, chunk_size = 100, overlap = 20) 
        texts = [chunk["text"] for chunk in chunks]
        
        # embedding
        embeddings =(self.embedding_service.embed_texts(texts))

        # store
        self.vector_store.save(
            embeddings=embeddings,
            metadata=chunks
        )