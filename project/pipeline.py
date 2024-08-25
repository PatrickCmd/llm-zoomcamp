import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from chunking import TranscriptChunker
from indexing import ChunkIndexer, ElasticsearchIndexingStrategy
from project.llm_rag import LLMRAGHandler
from transcript_cleaner import TranscriptCleaner
from transcription import YouTubeVideoTranscript

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_current_timestamp():
    """
    Generates a formatted timestamp string suitable for use in filenames.

    The timestamp is based on the current date and time and is formatted
    in the 'YYYYMMDD_HHMMSS' format, which ensures that it can be safely
    used in filenames without containing special characters.

    Returns:
        str: A string representing the current timestamp formatted as 'YYYYMMDD_HHMMSS'.
    """
    # Get the current time
    current_time = datetime.now()
    # Format the timestamp to remove special characters
    formatted_timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    return formatted_timestamp


def main(playlist_url, index_name="youtube_transcripts"):
    cleaner = TranscriptCleaner(model="gpt-3.5-turbo")
    youtube_tanscript = YouTubeVideoTranscript(
        api_key=os.getenv("YOUTUBE_DATA_API_KEY")
    )
    transcripts = youtube_tanscript.process_playlist(playlist_url, cleaner=cleaner)
    print(f"Number of processed transcripts: {len(transcripts)}")
    # Save transcripts to file
    current_timestamp = get_current_timestamp()
    file_name = f"data/real_python_podcast_{current_timestamp}.json"
    with open(file_name, "w") as file:
        json.dump(transcripts, file)

    chunker = TranscriptChunker(strategy="intelligent")
    es_strategy = ElasticsearchIndexingStrategy(index_name=index_name)

    indexer = ChunkIndexer(strategy=es_strategy)

    for _, transcript in transcripts.items():
        chunks = chunker.chunk_transcript(transcript)
        # Index the embeddings along with their corresponding chunks
        embeddings = indexer.create_embeddings(chunks)
        indexer.build_index(embeddings, chunks)

    return indexer


if __name__ == "__main__":
    playlist_url = (
        "https://www.youtube.com/playlist?list=PLP8GkvaIxJP0zDf2TqlGkqlBOjIuzm_BJ"
    )
    index_name = "the_real_python_podcast_transcripts"
    # indexer = main(playlist_url, index_name=index_name)

    es_strategy = ElasticsearchIndexingStrategy(index_name=index_name)
    indexer = ChunkIndexer(strategy=es_strategy)

    # Testing out querying the llm rag pipeline
    llm_rag = LLMRAGHandler(chunk_indexer=indexer, client=openai_client)
    query = "In the first episode of `The Real Python Podcast`, what were the key concepts discussed?"

    response, tokens, response_time = llm_rag.rag(query, k=5)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the fourth episode of `The Real Python Podcast`, who was the host and who were his co-hosts? What were the key concepts discussed?"
    response, tokens, response_time = llm_rag.rag(query, k=5)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the fourth episode of `The Real Python Podcast`, list atleast fifteen main topics and key concepts that were the focus of discussion?"
    response, tokens, response_time = llm_rag.rag(query, k=5)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the thirty third episode of `The Real Python Podcast`, list atleast twenty main topics and key concepts that were the focus of discussion?"
    response, tokens, response_time = llm_rag.rag(query, k=5)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "List episodes where the Django python framework was among the topics talked about"
    response, tokens, response_time = llm_rag.rag(query, k=15)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "List episodes where the topics that were discussed follow in the domains of Data, AI, Machine Learning (ML) and Data Engineering?"
    response, tokens, response_time = llm_rag.rag(query, k=15)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)
