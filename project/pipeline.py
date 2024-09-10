import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

from chunking import TranscriptChunker
from indexing import ChunkIndexer, ElasticsearchIndexingStrategy
from llm_rag import LLMRAGHandler
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


def load_real_python_podcast_json(directory_path):
    """
    Reads all JSON files in the specified directory that begin with 'real_python_podcast'
    and updates a single dictionary with the contents of those JSON files.

    This function iterates through all the files in the provided directory, identifies
    JSON files that start with 'real_python_podcast', and reads their content. The contents
    of each JSON file are merged into a single dictionary (`podcast_data`), ensuring that
    the data from each file is added directly to the dictionary.

    Parameters:
    ----------
    directory_path : str
        The path to the directory containing the files.

    Returns:
    -------
    dict
        A dictionary containing the merged contents of all the 'real_python_podcast' JSON files.
        If multiple JSON files have the same keys, later files will overwrite the values from earlier ones.

    Raises:
    ------
    FileNotFoundError:
        If the specified directory does not exist.

    Examples:
    --------
    >>> directory = '/path/to/json/files'
    >>> podcast_data = load_real_python_podcast_json(directory)
    >>> print(podcast_data)
    {
        'episode_1_data_key1': 'transcript1',
        'episode_1_data_key2': 'transcript2',
        'episode_2_data_key1': 'transcriptA',
        ...
    }
    """

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")

    podcast_transcripts_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file starts with 'real_python_podcast' and has a '.json' extension
        if filename.startswith("real_python_podcast") and filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            # Open and load the JSON file
            with open(file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                    # Directly update podcast_data with the JSON file's content
                    podcast_transcripts_data.update(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {filename}")

    return podcast_transcripts_data


def main(
    playlist_url,
    index_name="youtube_transcripts",
    scheme="http",
    host="localhost",
    port="9200",
    load_json_transcripts=True,
):
    cleaner = TranscriptCleaner(model="gpt-3.5-turbo")
    youtube_tanscript = YouTubeVideoTranscript(
        api_key=os.getenv("YOUTUBE_DATA_API_KEY")
    )
    if load_json_transcripts:
        print(f"Loading real python json transcripts data")
        transcripts = load_real_python_podcast_json("./data")
        print(f"Number of processed transcripts: {len(transcripts)}")
    else:
        transcripts = youtube_tanscript.process_playlist(playlist_url, cleaner=cleaner)
        print(f"Number of processed transcripts: {len(transcripts)}")
        # Save transcripts to file
        current_timestamp = get_current_timestamp()
        file_name = f"data/real_python_podcast_{current_timestamp}.json"
        with open(file_name, "w") as file:
            json.dump(transcripts, file)

    chunker = TranscriptChunker(strategy="intelligent")
    es_strategy = ElasticsearchIndexingStrategy(
        index_name=index_name, scheme=scheme, host=host, port=port
    )

    indexer = ChunkIndexer(strategy=es_strategy)

    for _, transcript in tqdm(transcripts.items()):
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
    indexer = main(playlist_url, index_name=index_name, load_json_transcripts=False)

    es_strategy = ElasticsearchIndexingStrategy(index_name=index_name)
    indexer = ChunkIndexer(strategy=es_strategy)

    # Testing out querying the llm rag pipeline
    llm_rag = LLMRAGHandler(
        chunk_indexer=indexer, client=openai_client, model="gpt-4o-mini"
    )
    query = "In the first episode of `The Real Python Podcast`, what were the key concepts discussed?"

    response, tokens, response_time = llm_rag.search(query, k=10)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the fourth episode of `The Real Python Podcast`, who was the host and who were his co-hosts? What were the key concepts discussed?"
    response, tokens, response_time = llm_rag.search(query, k=10)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the fourth episode of `The Real Python Podcast`, list atleast fifteen main topics and key concepts that were the focus of discussion?"
    response, tokens, response_time = llm_rag.search(query, k=10)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "In the thirty third episode of `The Real Python Podcast`, list atleast twenty main topics and key concepts that were the focus of discussion?"
    response, tokens, response_time = llm_rag.search(query, k=10)
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
    print(f"Response time: {response_time}")
    print("=" * 100)

    query = "List episodes where the Django python framework was among the topics talked about"
    response, tokens, response_time = llm_rag.search(query, k=15)
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
