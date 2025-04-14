import asyncio
import re
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from llm_rag import embeddings, llm, project_path
from pytube import Playlist
from pytube.innertube import _default_clients

VECTORSTORE_PATH = project_path / "data/vectorstore/chroma"


async def load_youtube_video_transcript(video_url):
    data = await YoutubeLoader.from_youtube_url(video_url, add_video_info=True).aload()
    return data[0]


async def load_documents():
    # https://github.com/pytube/pytube/issues/1894#issue-2180600881
    _default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"

    playlist = Playlist(
        "https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x"
    )
    coros = [
        load_youtube_video_transcript(video_url) for video_url in playlist.video_urls
    ]
    docs = await asyncio.gather(*coros)
    return docs


def load_data():
    return asyncio.run(load_documents())


def generate_chunk_content(chunk):
    return "\n\n".join(
        [
            f"Title:\n{chunk.metadata['title']}",
            f"Description:\n{chunk.metadata['description']}",
            f"Transcript:\n{chunk.page_content}",
        ]
    )


def prepare_vectorstore(docs, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    datetime_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y%m%d"
    processed_splits = []

    for split in splits:
        processed_split = split.copy()
        processed_split.metadata["publish_date"] = int(
            datetime.strptime(
                processed_split.metadata["publish_date"], datetime_format
            ).strftime(date_format)
        )
        processed_split.page_content = generate_chunk_content(processed_split)
        processed_splits.append(processed_split)

    collection_name = "youtube-rag-from-scratch"
    vectorstore_settings = Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_PATH), settings=vectorstore_settings
    )
    Chroma(collection_name=collection_name, client=client).delete_collection()
    vectorstore = Chroma(
        collection_name=collection_name, embedding_function=embeddings, client=client
    )
    vectorstore.add_documents(documents=processed_splits)

    return vectorstore


def generate_query_constructor_prompt():
    translator = ChromaTranslator()
    document_content_description = "Tutorial videos about RAG"
    metadata_field_info = [
        AttributeInfo(
            name="view_count",
            description="Video views count",
            type="integer",
        ),
        AttributeInfo(
            name="publish_date",
            description="Video publish date in format YYYYMMDD",
            type="int",
        ),
        AttributeInfo(
            name="length",
            description="Video length (seconds)",
            type="float",
        ),
    ]
    examples = [
        (
            "Find videos under 5 minutes",
            {
                "query": "Videos with length less than 300 seconds",
                "filter": 'lt("length", 300.0)',
            },
        ),
        (
            "Find videos published in 2024",
            {
                "query": "Videos with date greater or equal than 2024-01-01 and less than 2025-01-01",
                "filter": 'and(gte("publish_date", 20240101), lt("publish_date", 20250101))',
            },
        ),
        (
            "Find videos about indexing",
            {
                "query": "Videos about indexing",
                "filter": "NO_FILTER",
            },
        ),
        (
            "Find 3 videos about indexing",
            {
                "query": "3 videos about indexing",
                "filter": "NO_FILTER",
                "limit": 3,
            },
        ),
    ]
    query_constructor_prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        examples=examples,
        allowed_comparators=translator.allowed_comparators,
        allowed_operators=translator.allowed_operators,
        enable_limit=True,
    )
    return query_constructor_prompt


def clean_json_string(message):
    pattern = r".*?```json\s*(.*?)\s*```"
    cleaned_string = re.sub(
        pattern, r"\1", message.content, flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned_string.strip()


@chain
def query_constructor(query):
    query_constructor_prompt = generate_query_constructor_prompt()
    query_constructor_prompt_messages = query_constructor_prompt.format(query=query)
    response = llm.invoke([HumanMessage(content=query_constructor_prompt_messages)])
    clean_response = clean_json_string(response)

    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=translator.allowed_comparators,
        allowed_operators=translator.allowed_operators,
    )
    parsed_response = output_parser.invoke(clean_response)

    return parsed_response


def get_collection_size(vectorstore):
    try:
        collection_size = len(vectorstore.get()["ids"])
    except Exception:
        collection_size = 0

    return collection_size


docs = load_data()
vectorstore = prepare_vectorstore(docs, embeddings)
translator = ChromaTranslator()
retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=translator,
    verbose=True,
    search_kwargs={"k": get_collection_size(vectorstore)},
)
