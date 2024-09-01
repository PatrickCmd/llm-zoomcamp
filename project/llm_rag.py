import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from indexing import ChunkIndexer, ElasticsearchIndexingStrategy

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class LLMRAGHandler:
    def __init__(self, chunk_indexer, client, model="gpt-3.5-turbo"):
        self.chunk_indexer = chunk_indexer
        self.client = client
        self.model = model

    def generate_prompt(self, query, retrieved_chunks):
        # Create a well-structured prompt for the LLM
        context = "\n\n".join(
            [f"{chunk['text'].strip()}" for _, chunk in enumerate(retrieved_chunks)]
        )
        prompt = (
            f"Question: {query}\n\n"
            f"The following are excerpts from a series of youtube video transcripts that may contain relevant information about `The Real Python Podcast`:\n\n"
            f"{context}\n\n"
            "Based on the above information, provide a detailed and accurate answer to the question."
        )
        return prompt

    def llm(self, prompt):
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()
        tokens = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        end_time = time.time()
        response_time = end_time - start_time

        return answer, tokens, response_time

    def rag(self, query, k=5):
        # query_embedding = self.chunk_indexer.model.encode([query])[0]
        retrieved_chunks = self.chunk_indexer.search(query, k)

        prompt = self.generate_prompt(query, retrieved_chunks)
        response, tokens, response_time = self.llm(prompt)

        return response, tokens, response_time


if __name__ == "__main__":
    # Initialize the LLMRAGHandler with the chunk indexer
    es_strategy = ElasticsearchIndexingStrategy(index_name="youtube_transcripts")

    # Index the embeddings along with their corresponding chunks
    indexer = ChunkIndexer(strategy=es_strategy)
    llm_query_handler = LLMRAGHandler(chunk_indexer=indexer, client=openai_client)

    # Handle a query
    queries = [
        "What is the background of the guest on this episode of the Real Python Podcast?",
        "How did the guest get involved with Python and Real Python?",
        "What was the guest's experience with Python before joining Real Python?",
        "Can you describe the guest's involvement in the Real Python publishing process?",
        "What topics were discussed during the guest's talk at PyCon?",
        "What are some projects the guest has been working on recently, including any related to the Raspberry Pi?",
        "What Python tools or libraries does the guest prefer for working interactively?",
        "What is the guest’s opinion on the potential of the Raspberry Pi?",
        "What are the main stages of the Real Python publishing process as described by the guest?",
        "What PyPI packages has the guest been working on, and what are their purposes?",
    ]
    for query in queries:
        response, tokens, response_time = llm_query_handler.rag(query, k=5)
        print(f"Response: {response}")
        print(f"Tokens: {tokens}")
        print(f"Response time: {response_time}")
        print("=" * 100)
