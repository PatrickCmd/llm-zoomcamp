import streamlit as st
from openai import OpenAI
import json
import re
import requests

st.title("Sunbird AI Translation Assistant")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
sb_api_auth_token = st.secrets["SB_API_AUTH_TOKEN"]

language_codes = {
    "English": "eng",
    "Luganda": "lug",
    "Runyankole": "nyn",
    "Acholi": "ach",
    "Ateso": "teo",
    "Lugbara": "lgg",
}


def get_language(language_codes, code):
    """
    Retrieve the language name corresponding to a given language code.

    Parameters:
    language_codes (dict): A dictionary mapping language names to their respective codes.
                           Example: {
                               "English": "eng",
                               "Luganda": "lug",
                               "Runyankole": "nyn",
                               "Acholi": "ach",
                               "Ateso": "teo",
                               "Lugbara": "lgg"
                           }
    code (str): The language code for which the corresponding language name is sought.
                Example: "eng", "lug", etc.

    Returns:
    str: The language name corresponding to the given code. If the code is not found,
         returns None.

    Example:
    >>> language_codes = {
    ...     "English": "eng",
    ...     "Luganda": "lug",
    ...     "Runyankole": "nyn",
    ...     "Acholi": "ach",
    ...     "Ateso": "teo",
    ...     "Lugbara": "lgg"
    ... }
    >>> get_language(language_codes, "eng")
    'English'
    >>> get_language(language_codes, "lug")
    'Luganda'
    >>> get_language(language_codes, "xyz")
    None
    """
    for language, lang_code in language_codes.items():
        if lang_code == code:
            return language
    return None  # Return None if the code is not found


def extract_dictionary_from_response(response):
    # Use regular expression to find the JSON part of the string
    match = re.search(r"{.*}", response, re.DOTALL)
    if match:
        json_part = match.group(0)
        # Convert the JSON string to a dictionary
        return json.loads(json_part)
    else:
        return None


def translate(text, source_language, target_language):
    url = "https://api.sunbird.ai/tasks/nllb_translate"
    token = sb_api_auth_token
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = {
        "source_language": source_language,
        "target_language": target_language,
        "text": text,
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()["output"].get("translated_text")


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content


def get_translation_task(context):
    messages = context.copy()
    new_context = {
        "role": "system",
        "content": """
        create a json summary of the previous translation request. The json summary should be in json block, just return it as a dictionary
        The dictionary below represents the language codes available now for the translate endpoint

        language_codes: {
            "English": "eng",
            "Luganda": "lug",
            "Runyankole": "nyn",
            "Acholi": "ach",
            "Ateso": "teo",
            "Lugbara": "lgg"
        }

        Extract the source_language, text_to_translate, and target_language and return the response as the example below

        {
            "source_language": lug,
            "target_language": "eng",
            "text": text_to_translate
        }
        """,
    }
    messages.append(new_context)

    response = get_completion_from_messages(messages, temperature=0)
    response = response.replace("```", "").replace("json", "")
    response = extract_dictionary_from_response(response)
    # st.write(response)
    return response


context = [
    {
        "role": "system",
        "content": """
        You are Sunbird AI Translation Bot, an automated service to translate between English and Ugandan languages, from Uganda local language to English, or between Ugandan languages themselves (From one local language to another local language). 
        You first greet the user, then collect the language the user's text is in (the source language), then ask for the text they want to translate, and finally ask for the target language they want to translate the text to. 
        Inform the users that the available languages for translation are English, Luganda, Acholi, Ateso, Lugbara, and Runyankole.
        The target language cannot be the same as the source language.

        If a user types in something not related to translation, let them know that you are just a translation assistant and cannot help with that, and give them proper instructions.

        You wait to collect the entire user request, then summarize it and check if the user is now ready to proceed with the translation, and tell them to press the translate button.
        Finally, you get their entire request, that is, the source language, the text to translate, and the target language to translate to.
        You respond in a short, very conversational, friendly style.
        The available languages for translation include:
        Source languages: English, Luganda, Acholi, Ateso, Lugbara, Runyankole
        Target languages: English, Luganda, Acholi, Ateso, Lugbara, Runyankole

        Remember, a user cannot give a target language that is the same as the source language.
        """,
    }
]


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = context

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you with translation today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Translate button
if st.button("Translate"):
    with st.spinner("Translation in process..."):
        translation_result = get_translation_task(st.session_state.messages)
        translated_text = translate(
            translation_result["text"],
            translation_result["source_language"],
            translation_result["target_language"],
        )
        translation_result["translated_text"] = translated_text
        source_language = get_language(
            language_codes, translation_result["source_language"]
        )
        target_language = get_language(
            language_codes, translation_result["target_language"]
        )
        response_message = (
            f"Translation completed from {source_language} "
            f"to {target_language}: {translated_text}."
        )
        with st.chat_message("assistant"):
            st.write(response_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_message}
            )
