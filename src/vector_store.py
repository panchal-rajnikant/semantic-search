import json
from pathlib import Path

import numpy as np

class VectorStore:

    def __init__(self, storage_dir: str = "storage"):
        self.items = [],
        self.storage_dir = Path(storage_dir)

        self.embeddings_path = (
            self.storage_dir / "embeddings.npy"
        )

        self.metadata_path = (
            self.storage_dir / "metadata.json"
        )

        self.embeddings = None
        self.metadata = []

    def save(self, embeddings, metadata: list[dict]):
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.embeddings_path, embeddings )

        with self.metadata_path.open("w",encoding="utf-8") as file:
            json.dump(metadata,file, indent=2)

    def load(self) -> None:

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                "Embeddings index not found"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Metadata file not found"
            )

        self.embeddings = np.load(self.embeddings_path)

        with self.metadata_path.open("r",encoding="utf-8") as file:
            self.metadata = json.load(file)
