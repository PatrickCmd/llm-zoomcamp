import json
import os

import pandas as pd
import spacy
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

nlp = spacy.load("en_core_web_sm")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def list_real_python_podcast_json_files(directory):
    """
    Lists all Real python podcast JSON files in the specified directory.

    Args:
        directory (str): The path to the directory where JSON files are to be listed.

    Returns:
        list: A list of filenames (str) that have the '.json' extension and start with real_python_podcast within the directory.
    """
    try:
        # List all files in the given directory
        files = os.listdir(directory)
        # Filter and return only the files with '.json' extension
        json_files = [
            file
            for file in files
            if file.endswith(".json") and file.startswith("real_python_podcast")
        ]
        return json_files
    except FileNotFoundError:
        print(f"The directory '{directory}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def semantic_chunking_spacy_multisentence(transcript_text, chunk_size=1000):
    """
    Chunk the transcript_text text into smaller segments based on semantic boundaries using spaCy.

    :param transcript_text: The full transcript_text text to be chunked.
    :return: List of text chunks, each containing multiple sentences.
    """
    # Parse the text with spaCy
    doc = nlp(transcript_text)

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


def generate_questions(transcript_chunk):
    # Construct the prompt using the template
    prompt_template = f"""
    Given the following transcript chunk from an episode of The Real Python Podcast:

    {transcript_chunk}

    Generate 5 detailed questions that could be asked based on the content discussed in this episode. 
    Ensure that the questions cover a variety of topics such as the guest's or host's background, technical discussions, 
    Python-related projects, and any notable events or personal anecdotes mentioned. Make sure that each generated 
    question has the episode number mentioned.

    Please provide the questions in a Python list format, like this:
    [
      "What is the guest's background in Python programming as mentioned in episode one?",
      "What projects has the guest been working on recently as related to episode ten?",
      "What were the main topics discussed during the host's talk at PyCon in Real Python podcast episode thirty?",
      "What specific Python tools or libraries were mentioned in the episode three?",
      "What future projects or goals did the guest mention during the episode eight?",
      ...
    ]
    """

    # Use the OpenAI API to generate the questions
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_template}],
        temperature=0.7,
    )
    # Extract and return the generated questions
    generated_questions = response.choices[0].message.content
    return generated_questions


def format_as_list(input_string):
    """
    Converts a string representation of a Python list into an actual Python list.

    This function takes a string that is expected to represent a Python list (with square brackets)
    and converts it into a proper Python list. The function performs basic checks to ensure that
    the input string starts with '[' and ends with ']', and then uses `eval` to convert the string
    into a list.

    Parameters:
    ----------
    input_string : str
        A string that represents a list, where elements are enclosed in square brackets and separated
        by commas.

    Returns:
    -------
    list
        The input string converted to a Python list.

    Raises:
    ------
    ValueError:
        If the input string does not start with '[' and end with ']'.

    Examples:
    --------
    >>> input_string = '''
    [
        "What is Python?",
        "How do you use decorators?",
        "What are some features of Python?"
    ]
    '''
    >>> format_as_list(input_string)
    ["What is Python?", "How do you use decorators?", "What are some features of Python?"]

    """
    # Remove any leading or trailing whitespace
    input_string = input_string.strip()

    # If the string starts and ends with square brackets, it's likely a list
    if input_string.startswith("[") and input_string.endswith("]"):
        # Convert the string to a Python list using eval
        formatted_list = eval(input_string)
        return formatted_list
    else:
        raise ValueError(
            "The provided string does not appear to be a properly formatted list."
        )


def load_transcripts(file_path):
    """
    Loads transcripts from a JSON file.

    This function checks if the specified file path exists. If the file exists,
    it reads the file and loads the contents as a dictionary using `json.load`.
    If the file does not exist, it returns an empty dictionary.

    Parameters:
    ----------
    file_path : str
        The path to the JSON file containing the transcripts.

    Returns:
    -------
    dict
        A dictionary containing the transcripts loaded from the file. If the
        file does not exist, an empty dictionary is returned.

    Examples:
    --------
    >>> transcripts = load_transcripts("transcripts.json")
    >>> print(transcripts)
    {'episode_1': 'Transcript text...', 'episode_2': 'Transcript text...'}
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}


def questions_list(file_path):
    """
    Generates a list of questions for each episode based on the transcripts.

    This function loads the transcripts from the specified JSON file, performs
    semantic chunking on the content of each episode, generates questions for
    each chunk, and formats them into a list. The function then associates each
    question with its corresponding episode ID.

    Parameters:
    ----------
    file_path : str
        The name of the file containing the transcripts (without the path).

    Returns:
    -------
    list of dict
        A list of dictionaries, each containing an 'episode_id' and a 'question'.
        Example:
        [{'episode_id': 'episode_1', 'question': 'What is Python?'},
         {'episode_id': 'episode_1', 'question': 'How do decorators work?'},
         ...]

    Examples:
    --------
    >>> questions = questions_list("transcripts.json")
    >>> print(questions)
    [{'episode_id': 'episode_1', 'question': 'What is Python?'},
     {'episode_id': 'episode_1', 'question': 'How do decorators work?'},
     {'episode_id': 'episode_2', 'question': 'What is object-oriented programming?'}]
    """
    # Create a list to hold all the questions with episode ids
    file_path = f"data/{file_path}"
    transcripts = load_transcripts(file_path)
    questions_list = []

    # Iterate over the data and generate questions for each episode
    for episode_id, content in transcripts.items():
        chunks = semantic_chunking_spacy_multisentence(content)
        for chunk in chunks:
            episode_questions = generate_questions(chunk)
            questions = format_as_list(episode_questions)
            for question in questions:
                questions_list.append({"episode_id": episode_id, "question": question})

    return questions_list


def real_python_questions_dataframe(questions_list):
    """
    Converts a list of questions into a DataFrame and saves it as a CSV file.

    This function takes a list of dictionaries containing episode IDs and their corresponding questions,
    converts the list into a Pandas DataFrame, and then saves the DataFrame as a CSV file named
    'real_python_podcast_questions.csv'. The function returns the DataFrame for further use.

    Parameters:
    ----------
    questions_list : list of dict
        A list where each dictionary contains 'episode_id' and 'question' as keys.

    Returns:
    -------
    pd.DataFrame
        A Pandas DataFrame containing the questions and episode IDs.

    Examples:
    --------
    >>> questions = [{'episode_id': 'episode_1', 'question': 'What is Python?'},
                     {'episode_id': 'episode_2', 'question': 'How do decorators work?'}]
    >>> df = real_python_questions_dataframe(questions)
    >>> print(df)
       episode_id                      question
    0  episode_1                What is Python?
    1  episode_2        How do decorators work?

    The resulting CSV file 'real_python_podcast_questions.csv' will have the same content.
    """
    # Convert to a DataFrame
    df = pd.DataFrame(questions_list)

    # Save as CSV
    output_file = "data/real_python_podcast_questions.csv"
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    real_python_podcast_json_files = list_real_python_podcast_json_files("data/")
    print(real_python_podcast_json_files)

    """
    text = ""
    with open("transcript.txt", "r") as tf:
        text = tf.read().strip()

    generated_questions = generate_questions(text)
    questions = format_as_list(generated_questions)
    print(questions)
    print(type(questions))
    """

    generated_questions_list = []
    for file in real_python_podcast_json_files:
        generated_questions_list = questions_list(file)

    print(f"Generated questions: {len(generated_questions_list)}")

    df = real_python_questions_dataframe(generated_questions_list)
    print(df.head(10))
