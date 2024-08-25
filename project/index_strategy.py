from abc import ABC, abstractmethod


class IndexingStrategy(ABC):
    @abstractmethod
    def build_index(self, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, k=5):
        pass
