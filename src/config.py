from dataclasses import dataclass
from typing import Literal

RetrivalStrategy = [
    "dense",
    "sparse",
    "hybrid"
]
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

    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    enable_rerankingTool:bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str = "hybrid"

    top_k: int = 3
    candidate_k: int = 10

    similarity_threshold: float = 0.40

    dense_weight: float = 0.5

    reranking_enabled: bool = True