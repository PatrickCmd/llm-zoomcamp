import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class TranscriptCleaner:
    """
    A class to clean and process auto-generated transcripts from YouTube videos using Large Language Models (LLMs).

    Methods
    -------
    clean_transcript(transcript: str) -> str:
        Cleans the transcript by removing filler words, correcting grammar, and simplifying sentences using an LLM.

    refine_transcript(transcript: str) -> str:
        Refines the transcript by ensuring accuracy, enhancing context, and formatting the text using an LLM.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initializes the TranscriptCleaner with the specified LLM model.

        Parameters
        ----------
        model : str
            The LLM model to use for cleaning and refining transcripts (e.g., "gpt-4", "gpt-3.5-turbo").
        """
        self.model = model

    def clean_transcript(self, transcript: str):
        """
        Cleans the transcript by using an LLM to remove filler words, correct grammar, and simplify sentences.

        Parameters
        ----------
        transcript : str
            The raw transcript as a string.

        Returns
        -------
        str
            The cleaned transcript.
        """
        prompt = f"""
        You are given a raw transcript of a conversation. Please clean the transcript by:
        1. Removing filler words like "um," "uh," and "you know."
        2. Correcting any grammatical errors.
        3. Format the transcript with proper paragraph breaks .

        Transcript: {transcript}

        Cleaned Transcript:
        """

        start_time = time.time()
        response = openai_client.chat.completions.create(
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

    def refine_transcript(self, transcript: str):
        """
        Refines the transcript by using an LLM to ensure accuracy, enhance context, and format the text.

        Parameters
        ----------
        transcript : str
            The cleaned transcript as a string.

        Returns
        -------
        str
            The refined transcript.
        """
        prompt = f"""
        Please refine the following transcript:
        1. Ensure all technical terms, names, and specific references are correct.
        2. Adjust the context where needed for clarity.
        3. Format the transcript with proper paragraph breaks and emphasize key points.

        Transcript: {transcript}

        Refined Transcript:
        """

        start_time = time.time()
        response = openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
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


if __name__ == "__main__":
    # Initialize the TranscriptCleaner with the desired LLM model
    cleaner = TranscriptCleaner(model="gpt-3.5-turbo")

    # Example raw transcript
    transcript = """
        Um, so this is the intro to the video. As I mentioned earlier, we will be discussing important aspects of Python programming.
        I think... well, it's critical to understand these concepts.
        Outro, thank you for watching!
    """

    # Clean the transcript using the LLM
    cleaned_transcript, _, _ = cleaner.clean_transcript(transcript)
    print("Cleaned Transcript:\n", cleaned_transcript)

    # Refine the cleaned transcript for accuracy and formatting
    refined_transcript, _, _ = cleaner.refine_transcript(cleaned_transcript)
    print("Refined Transcript:\n", refined_transcript)
