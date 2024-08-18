from sentence_transformers import SentenceTransformer
import faiss
from elasticsearch import Elasticsearch, helpers
import numpy as np

from index_strategy import IndexingStrategy


class FaissIndexingStrategy(IndexingStrategy):
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def build_index(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices


class ElasticsearchIndexingStrategy(IndexingStrategy):
    def __init__(self, index_name='youtube_transcripts', host='localhost', port=9200):
        self.es = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = index_name
        self._create_index()

    def _create_index(self):
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384  # Change this to the actual dimension of your embeddings
                            },
                            "text": {
                                "type": "text"
                            }
                        }
                    }
                }
            )

    def build_index(self, embeddings, chunks):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "embedding": embedding.tolist(),
                    "text": chunk
                }
            }
            for embedding, chunk in zip(embeddings, chunks)
        ]
        helpers.bulk(self.es, actions)

    def search(self, query_embedding, k=5):
        query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding.tolist()
                        }
                    }
                }
            }
        }
        response = self.es.search(index=self.index_name, body=query)
        hits = response['hits']['hits']
        results = [{"text": hit['_source']['text'], "score": hit['_score']} for hit in hits]
        return results


class ChunkIndexer:
    def __init__(self, strategy: IndexingStrategy, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.strategy = strategy

    def create_embeddings(self, chunks):
        embeddings = self.model.encode(chunks)
        return embeddings

    def build_index(self, embeddings):
        self.strategy.build_index(embeddings)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        return self.strategy.search(query_embedding, k)


if __name__ == "__main__":
    # Define the text chunks (example text)
    chunks = [
        "This is the first chunk of text from the video transcript.",
        "This is the second chunk of text, discussing another topic.",
        "Here we talk about some important concepts related to the video.",
        "Final thoughts and conclusions are provided in this chunk."
    ]

    # FAISS Example
    faiss_strategy = FaissIndexingStrategy(dimension=384)  # Assuming the embedding dimension is 384
    indexer = ChunkIndexer(strategy=faiss_strategy)
    embeddings = indexer.create_embeddings(chunks)
    indexer.build_index(embeddings)

    # Elasticsearch Example

    # Initialize the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings for the chunks
    embeddings = model.encode(chunks)

    # Initialize the ElasticsearchIndexingStrategy
    es_strategy = ElasticsearchIndexingStrategy(index_name='youtube_transcripts')

    # Index the embeddings along with their corresponding chunks
    indexer = ChunkIndexer(strategy=es_strategy)
    es_strategy.build_index(embeddings, chunks)

    # Query Elasticsearch with a sample query and retrieve the top 5 results
    query = "What are the important concepts discussed?"
    query_embedding = model.encode([query])[0]
    results = es_strategy.search(query_embedding, k=5)

    # Print the results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

