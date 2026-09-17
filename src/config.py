from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    chunk_size: int = 100
    chunk_overlap: int = 20

    embedding_model: str = (
        "all-MiniLM-L6-v2"
    )

    top_k: int = 3
    similarity_threshold: float = 0.40

    data_dir: str = "data"
    storage_dir: str = "storage"