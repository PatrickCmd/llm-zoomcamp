from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(find_dotenv())

project_path = Path(__file__).resolve().parents[2]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
