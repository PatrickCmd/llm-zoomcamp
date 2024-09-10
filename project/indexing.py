from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from chunking import TranscriptChunker
from index_strategy import IndexingStrategy


class ElasticsearchIndexingStrategy(IndexingStrategy):
    def __init__(
        self,
        index_name="youtube_transcripts",
        scheme="http",
        host="localhost",
        port=9200,
        delete_index=False,
    ):
        self.es = Elasticsearch([{"scheme": scheme, "host": host, "port": port}])
        self.index_name = index_name
        if delete_index:
            self.es.indices.delete(index=self.index_name, ignore_unavailable=True)
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
                                "dims": 384,  # Change this to the actual dimension of your embeddings
                            },
                            "text": {"type": "text"},
                        }
                    }
                },
            )

    def build_index(self, embeddings, chunks):
        actions = [
            {
                "_index": self.index_name,
                "_source": {"embedding": embedding.tolist(), "text": chunk},
            }
            for embedding, chunk in zip(embeddings, chunks)
        ]
        helpers.bulk(self.es, actions)

    def search(self, query_embedding, k=5):
        query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding.tolist()},
                    },
                }
            },
        }
        response = self.es.search(index=self.index_name, body=query)
        hits = response["hits"]["hits"]
        results = [
            {"text": hit["_source"]["text"], "score": hit["_score"]} for hit in hits
        ]
        return results


class ChunkIndexer:
    def __init__(
        self,
        strategy: IndexingStrategy,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    ):
        print("Initialize the SentenceTransformer model")
        self.model = SentenceTransformer(embedding_model)
        print(f"Model: {self.model}")
        self.strategy = strategy

    def create_embeddings(self, chunks):
        print("Create chunk embeddings")
        embeddings = self.model.encode(chunks)
        return embeddings

    def build_index(self, embeddings, chunks=[]):
        print(f"Index the embeddings along with their corresponding chunks")
        if chunks:
            self.strategy.build_index(embeddings, chunks)
        else:
            self.strategy.build_index(embeddings)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])[0]
        return self.strategy.search(query_embedding, k)


if __name__ == "__main__":
    chunker = TranscriptChunker(strategy="intelligent")

    # Example usage
    text = ""
    with open("transcript.txt", "r") as tf:
        text = tf.read().strip()

    # Chunk the text by paragraphs
    chunks = chunker.chunk_transcript(text)
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk: {chunks[0]}")

    # Elasticsearch Example

    # Initialize the SentenceTransformer model
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for the chunks
    # embeddings = model.encode(chunks)

    # Initialize the ElasticsearchIndexingStrategy
    es_strategy = ElasticsearchIndexingStrategy(index_name="youtube_transcripts")

    # Index the embeddings along with their corresponding chunks
    print("Index the embeddings along with their corresponding chunks")
    indexer = ChunkIndexer(strategy=es_strategy)
    embeddings = indexer.create_embeddings(chunks)
    indexer.build_index(embeddings, chunks)

    # Query Elasticsearch with a sample query and retrieve the top 5 results
    query = "What are the important concepts discussed?"
    # query_embedding = indexer.model.encode([query])[0]
    # print(query_embedding)
    results = indexer.search(query, k=5)

    # Print the results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")
