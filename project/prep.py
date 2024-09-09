import os

import pandas as pd
import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from db import init_db
from helpers import parse_url
from pipeline import main as index_processor

load_dotenv()

ELASTIC_URL = os.getenv(
    "ELASTIC_URL", "http://elasticsearch:9200"
)  # os.getenv("ELASTIC_URL_LOCAL")
MODEL_NAME = os.getenv("MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

playlist_url = (
    "https://www.youtube.com/playlist?list=PLP8GkvaIxJP0zDf2TqlGkqlBOjIuzm_BJ"
)
index_name = INDEX_NAME
scheme, host, port = parse_url(ELASTIC_URL)


"""
def fetch_ground_truth():
    print("Fetching ground truth data...")
    relative_url = "03-vector-search/eval/ground-truth-data.csv"
    ground_truth_url = f"{BASE_URL}/{relative_url}?raw=1"
    df_ground_truth = pd.read_csv(ground_truth_url)
    df_ground_truth = df_ground_truth[
        df_ground_truth.course == "machine-learning-zoomcamp"
    ]
    ground_truth = df_ground_truth.to_dict(orient="records")
    print(f"Fetched {len(ground_truth)} ground truth records")
    return ground_truth
"""


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def setup_elasticsearch():
    print("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL)
    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    return es_client


def load_index_podcast_transcripts(
    playlist_url, index_name, load_json_transcripts=True
):
    setup_elasticsearch()
    indexer = index_processor(
        playlist_url,
        index_name=index_name,
        load_json_transcripts=load_json_transcripts,
        scheme=scheme,
        host=host,
        port=port,
    )
    return indexer


def main():
    # print("Starting the indexing process...")

    load_index_podcast_transcripts(
        playlist_url=playlist_url, index_name=INDEX_NAME, load_json_transcripts=True
    )

    print("Initializing database...")
    init_db()

    print("Indexing process completed successfully!")


if __name__ == "__main__":
    main()
