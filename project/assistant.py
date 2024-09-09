import json
import os

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from helpers import parse_url
from indexing import ChunkIndexer, ElasticsearchIndexingStrategy
from llm_rag import LLMRAGHandler

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")


ollama_client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
index_name = os.getenv("INDEX_NAME", "the_real_python_podcast_transcripts")
model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

embedding_model = model_name

scheme, host, port = parse_url(ELASTIC_URL)
es_strategy = ElasticsearchIndexingStrategy(
    index_name=index_name, scheme=scheme, host=host, port=port
)
indexer = ChunkIndexer(strategy=es_strategy, embedding_model=embedding_model)


def llm(query, model_choice, top_k=5):
    if model_choice.startswith("ollama/"):
        model = model_choice.split("/")[-1]
        llm_rag = LLMRAGHandler(
            chunk_indexer=indexer, client=ollama_client, model=model
        )
        answer, tokens, response_time = llm_rag.search(query, k=top_k)
    elif model_choice.startswith("openai/"):
        model = model_choice.split("/")[-1]
        llm_rag = LLMRAGHandler(
            chunk_indexer=indexer, client=openai_client, model=model
        )
        answer, tokens, response_time = llm_rag.search(query, k=top_k)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    return answer, tokens, response_time


def evaluate_relevance(question, answer):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt, "openai/gpt-4o-mini")

    try:
        json_eval = json.loads(evaluation)
        return json_eval["Relevance"], json_eval["Explanation"], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(model_choice, tokens):
    openai_cost = 0

    if model_choice == "openai/gpt-3.5-turbo":
        openai_cost = (
            tokens["prompt_tokens"] * 0.0015 + tokens["completion_tokens"] * 0.002
        ) / 1000
    elif model_choice in ["openai/gpt-4o", "openai/gpt-4o-mini"]:
        openai_cost = (
            tokens["prompt_tokens"] * 0.03 + tokens["completion_tokens"] * 0.06
        ) / 1000

    return openai_cost


def get_answer(query, model_choice, top_k=5):

    answer, tokens, response_time = llm(query, model_choice, top_k=top_k)

    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    openai_cost = calculate_openai_cost(model_choice, tokens)

    return {
        "answer": answer,
        "response_time": response_time,
        "relevance": relevance,
        "relevance_explanation": explanation,
        "model_used": model_choice,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "eval_prompt_tokens": eval_tokens["prompt_tokens"],
        "eval_completion_tokens": eval_tokens["completion_tokens"],
        "eval_total_tokens": eval_tokens["total_tokens"],
        "openai_cost": openai_cost,
    }
