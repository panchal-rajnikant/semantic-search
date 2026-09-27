from .base import EmbeddingProvider, EmbeddingProviderError
from sentence_transformers import SentenceTransformer

class SentenceTransformerProvider(EmbeddingProvider):

    def __init__(self, model_name: str, batch_size: int):
        try:
            self._model_name = model_name
            self.batch_size = batch_size

            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingProviderError(f"failed to load embedding model: {model_name}") from exc
