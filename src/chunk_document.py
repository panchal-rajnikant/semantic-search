from src.chunker import chunk_text
def chunk_document(documents: list[dict],chunk_size:int = 100, overlap:int = 20)->list[dict]:
    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["text"],
            chunk_size,
            overlap
        )

        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": chunk
                }
            )

    return all_chunks