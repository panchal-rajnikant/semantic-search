import hashlib

def chunk_text(text:str, chunk_size:int = 100, overlap:int=20, strategy:str="fixed")->list[str]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    if strategy not in ("fixed", "recursive"):
        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )
    
    words = text.split()

    if not words:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size

        # chunk_words = words[start:end]
        chunk_words = words[start:end]

        if not chunk_words:
            continue


        chunks.append(
            " ".join(chunk_words)
        )
        
         # We reached the end of the document.
        if end >= len(words):
            break

    return chunks

def calculate_content_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def chunk_documents(
    documents: list[dict],
    chunk_size: int,
    overlap: int,
    strategy: str
) -> list[dict]:

    all_chunks = []

    for document in documents:
        content_hash = calculate_content_hash(document["text"])

        chunks = chunk_text(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy
        )

        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append({
                "document_id": document["document_id"],
                "source": document["source"],
                "content_hash": content_hash,
                "chunk_id": chunk_id,
                "text": chunk
            })

    return all_chunks