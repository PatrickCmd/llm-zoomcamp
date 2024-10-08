import json
import os
import re
import sys

import spacy
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

from transcript_cleaner import TranscriptCleaner

load_dotenv()


class YouTubeVideoTranscript:
    def __init__(
        self, api_key: str, processed_ids_file="data/processed_video_ids.json"
    ):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        self.processed_ids_file = processed_ids_file
        self.processed_video_ids = self.load_processed_video_ids()
        print(f"Processed video ids: {len(self.processed_video_ids)}")
        self.nlp = spacy.load("en_core_web_sm")

    def load_processed_video_ids(self):
        if os.path.exists(self.processed_ids_file):
            with open(self.processed_ids_file, "r") as file:
                return json.load(file)
        return {}

    def save_processed_video_ids(self):
        with open(self.processed_ids_file, "w") as file:
            json.dump(self.processed_video_ids, file)

    def get_video_ids(self, playlist_url: str):
        # Extract playlist ID from the URL
        playlist_id = self.extract_playlist_id(playlist_url)

        video_ids = []
        next_page_token = None

        # Paginate through the playlist items to get all video IDs
        while True:
            request = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            response = request.execute()

            # Collect video IDs from the response
            video_ids.extend(
                [item["contentDetails"]["videoId"] for item in response["items"]]
            )

            # Check if there is another page of results
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids[::-1]

    def extract_playlist_id(self, playlist_url: str):
        # Use regex to extract the playlist ID from the URL
        match = re.search(r"list=([a-zA-Z0-9_-]+)", playlist_url)
        if not match:
            raise ValueError("Invalid YouTube playlist URL")
        return match.group(1)

    def semantic_chunking_spacy_multisentence(self, transcript_text, chunk_size=1000):
        """
        Chunk the transcript_text text into smaller segments based on semantic boundaries using spaCy.

        :param transcript_text: The full transcript_text text to be chunked.
        :return: List of text chunks, each containing multiple sentences.
        """
        # Parse the text with spaCy
        doc = self.nlp(transcript_text)

        # Initialize chunk variables
        chunks = []
        chunk = ""

        for sentence in doc.sents:
            if len(chunk.split()) + len(sentence.text.split()) > chunk_size:
                if len(chunk) != 0:
                    chunks.append(chunk)
                chunk = sentence.text
            else:
                chunk += " " + sentence.text

        # Add the last chunk if it's not empty
        if chunk:
            chunks.append(chunk)

        return chunks

    def get_transcript(self, video_id: str):
        # Use YouTubeTranscriptApi to get the transcript for each video (Default is JSON format)
        # Formatting it to text
        formatter = TextFormatter()
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text_formatted_transcript = formatter.format_transcript(transcript)
        return text_formatted_transcript

    def clean_transcript(self, transcript: str, cleaner):
        # Clean the transcript (e.g., remove timestamps, non-verbal sounds, etc.)
        chunks = self.semantic_chunking_spacy_multisentence(transcript_text=transcript)
        transcript = ""
        for chunk in chunks:
            cleaned_transcript, _, _ = cleaner.clean_transcript(chunk)

            # Refine the cleaned transcript for accuracy and formatting
            # refined_transcript, _, _ = cleaner.refine_transcript(cleaned_transcript)
            transcript += cleaned_transcript
        return transcript.strip()

    def process_playlist(self, playlist_url: str, cleaner):
        video_ids = self.get_video_ids(playlist_url)
        cleaned_transcripts = {}

        for video_id in video_ids:
            if video_id in self.processed_video_ids:
                print(f"Skipping video {video_id}, already processed.")
                continue

            try:
                raw_transcript = self.get_transcript(video_id)
                cleaned_transcript = self.clean_transcript(raw_transcript, cleaner)
                cleaned_transcripts[video_id] = cleaned_transcript

                # Mark this video as processed
                self.processed_video_ids[video_id] = True
            except Exception as e:
                print(f"Transcription Error: {str(e)}")

        # Save the processed video IDs
        self.save_processed_video_ids()

        return cleaned_transcripts


if __name__ == "__main__":
    api_key = os.getenv("YOUTUBE_DATA_API_KEY")
    youtube_playlist_url = (
        "https://www.youtube.com/playlist?list=PLP8GkvaIxJP0zDf2TqlGkqlBOjIuzm_BJ"
    )
    cleaner = TranscriptCleaner(model="gpt-3.5-turbo")

    youtube = YouTubeVideoTranscript(api_key=api_key)
    print(f"youtube: {youtube.youtube}")
    playlist_id = youtube.extract_playlist_id(youtube_playlist_url)
    print(f"playlist_id: {playlist_id}")
    video_ids = youtube.get_video_ids(youtube_playlist_url)
    print(f"Video IDs: {video_ids} Number of Videos: {len(video_ids)}")
    video_id = video_ids[0]  # aL5GK2LVMWI
    video_transcript = youtube.get_transcript(video_id)
    cleaned_transcript = youtube.clean_transcript(video_transcript, cleaner)
    print(f"Video transcript: {cleaned_transcript}")
    # youtube.process_playlist(youtube_playlist_url, cleaner)
