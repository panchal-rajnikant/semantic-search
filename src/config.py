from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    data_dir: str = "data"
    storage_dir: str = "storage"

    chunk_size: int = 500
    chunk_overlap: int = 50

    embedding_model: str = (
        "all-MiniLM-L6-v2"
    )

    top_k: int = 3
    similarity_threshold: float = 0.40
