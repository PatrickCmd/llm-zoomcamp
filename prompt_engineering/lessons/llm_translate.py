import os
import requests
import openai
import re
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Ensure your OpenAI API key is set as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# Dictionary for language codes
language_codes = {
    "English": "eng",
    "Luganda": "lug",
    "Runyankole": "nyn",
    "Acholi": "ach",
    "Ateso": "teo",
    "Lugbara": "lgg"
}

def translate_via_api(text, source_language, target_language):
    url = "https://api.sunbird.ai/tasks/nllb_translate"
    token = os.getenv("AUTH_TOKEN")
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
    # print(f"Response: {response.json()}")
    return response.json()["output"].get("translated_text")

def llm_translate(openai_model, user_text):
    # Extract the translation request from the user text
    match = re.match(r"Translate this text from (\w+) to (\w+): ```(.+?)```", user_text)
    if match:
        source_language_name, target_language_name, text_to_translate = match.groups()
        source_language = language_codes.get(source_language_name)
        target_language = language_codes.get(target_language_name)
        
        if source_language and target_language:
            # Define the prompt template
            prompt_template = """
            You are an assistant that helps translate text using a specific translation API.
            The available languages are English (eng), Luganda (lug), Runyankole (nyn), Acholi (ach), Ateso (teo), and Lugbara (lgg).
            Return a JSON object containing the source language, target language, original text, and translated text.
            
            Example input: "Translate this text from English to Luganda: ```Are we going for soccer today?```"
            Example output: {{"source_language": "eng", "target_language": "lug", "text": "Are we going for soccer today?", "translated_text": "Tugenda kuzannya mupiira leero?"}}
            
            Translate the following text from {source_language_name} to {target_language_name}: ```{text_to_translate}```
            """
            
            # Replace placeholders in the prompt template
            prompt = prompt_template.format(
                source_language_name=source_language_name,
                target_language_name=target_language_name,
                text_to_translate=text_to_translate
            )

            # Generate response using OpenAI model
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0
            )

            completion_text = response.choices[0].message.content.strip()

            try:
                translation_request = json.loads(completion_text)
                source_language = translation_request.get("source_language")
                target_language = translation_request.get("target_language")
                text_to_translate = translation_request.get("text")
                translated_text = translate_via_api(text_to_translate, source_language, target_language)
                if source_language and target_language and text_to_translate:
                    translated_text = translate_via_api(text_to_translate, source_language, target_language)
                    translation_request["translated_text"] = translated_text
                    return translation_request
                else:
                    return {"translated_text": "Invalid input format."}
            except Exception as e:
                return {"translated_text": str(e)}
        else:
            return {"translated_text": "Invalid language provided."}
    else:
        return {"translated_text": "Invalid input format."}

# Example usage
if __name__ == "__main__":
    user_text = "Translate this text from English to Runyankole: ```How are you feeling today evening?```"
    translated_response = llm_translate("gpt-3.5-turbo", user_text)
    print(translated_response)
