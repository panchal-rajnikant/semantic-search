from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self,query: str)-> np.ndarray:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass
