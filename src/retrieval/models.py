from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    document_id: str
    chunk_id: int
    metadata: dict[str, Any]
    rank: int | None = None