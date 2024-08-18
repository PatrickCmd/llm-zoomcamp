import os
import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class LLMQueryHandler:
    def __init__(self, chunk_indexer, client):
        self.chunk_indexer = chunk_indexer
        self.client = client

    def generate_prompt(self, query, retrieved_chunks):
        # Create a well-structured prompt for the LLM
        context = "\n\n".join([f"Chunk {i+1}: {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)])
        prompt = (
            f"Question: {query}\n\n"
            f"The following are excerpts from a series of video transcripts that may contain relevant information:\n\n"
            f"{context}\n\n"
            "Based on the above information, provide a detailed and accurate answer to the question."
        )
        return prompt

    def get_response(self, prompt):
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()
        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        end_time = time.time()
        response_time = end_time - start_time

        return answer, tokens, response_time

    def handle_query(self, query, k=5):
        query_embedding = self.chunk_indexer.model.encode([query])[0]
        retrieved_chunks = self.chunk_indexer.strategy.search(query_embedding, k)

        prompt = self.generate_prompt(query, retrieved_chunks)
        response = self.get_response(prompt)

        return response


if __name__ == "__main__":
    # Initialize the LLMQueryHandler with the chunk indexer
    llm_query_handler = LLMQueryHandler(chunk_indexer=indexer, openai_api_key=openai_api_key)

    # Handle a query
    query = "What are the key concepts discussed in these transcripts?"
    response = llm_query_handler.handle_query(query, k=5)

    # Print the response
    print(response)

