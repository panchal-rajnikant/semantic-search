import numpy as np

from src.loader import load_documents
from src.chunker import chunk_documents, calculate_content_hash
from src.config import SearchConfig


class IngestionService:

    def __init__(self, EmbeddingService, vector_store):
        self.embedding_service = EmbeddingService
        self.vector_store = vector_store
        self.config = SearchConfig()

    def ingest(self, data_dir: str) -> None:

        # 1. Load documents
        documents = load_documents(data_dir)

        # 2. Load the saved index before checking document hashes
        if (
            self.vector_store.embeddings_path.exists()
            or self.vector_store.metadata_path.exists()
        ):
            self.vector_store.load()

        existing_metadata = self.vector_store.metadata

        existing_hashes = {
            item["document_id"]: item.get("content_hash")
            for item in existing_metadata
        }
        existing_document_ids = set(existing_hashes)
        current_document_ids = {
            document["document_id"] for document in documents
        }

        documents_to_ingest = []
        unchanged_document_ids = set()

        # 3. Detect new, unchanged and updated documents
        for document in documents:

            document_id = document["document_id"]
            new_hash = calculate_content_hash(document["text"])
            existing_hash = existing_hashes.get(document_id)

            if document_id in existing_hashes and existing_hash == new_hash:
                print(f"Skipping unchanged document: {document_id}")
                unchanged_document_ids.add(document_id)
                continue

            documents_to_ingest.append(document)

        removed_document_ids = existing_document_ids - current_document_ids

        # 4. Stop if nothing changed
        if not documents_to_ingest and not removed_document_ids:
            print("No new or updated documents.")
            return

        # 5. Chunk only new or updated documents
        chunks = chunk_documents(
            documents=documents_to_ingest,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )

        # 6. Generate embeddings
        new_embeddings = None
        if chunks:
            texts = [chunk["text"] for chunk in chunks]
            new_embeddings = self.embedding_service.embed_texts(texts)

        # 7. Keep embeddings for unchanged documents and replace changed ones
        retained_indices = [
            index
            for index, item in enumerate(existing_metadata)
            if item["document_id"] in unchanged_document_ids
        ]
        retained_metadata = [
            item
            for item in existing_metadata
            if item["document_id"] in unchanged_document_ids
        ]

        if self.vector_store.embeddings is not None:
            retained_embeddings = self.vector_store.embeddings[retained_indices]
        else:
            retained_embeddings = None

        if new_embeddings is None:
            embeddings = retained_embeddings
        elif retained_embeddings is None or not len(retained_metadata):
            embeddings = new_embeddings
        else:
            embeddings = np.concatenate(
                (retained_embeddings, new_embeddings),
                axis=0
            )

        if embeddings is None:
            raise RuntimeError("Cannot save an index without embeddings.")

        self.vector_store.save(
            embeddings=embeddings,
            metadata=retained_metadata + chunks
        )