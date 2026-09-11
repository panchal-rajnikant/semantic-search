from pathlib import Path
from src.chunker import chunk_text

def load_documents(data_dir:str)->list[dict]:
    
    documents = []

    directory = Path(data_dir)

    for file_path in directory.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            continue
        
        document = {
            "source": file_path.name,
            "text": text
        }

        documents.append(document)

    return documents

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