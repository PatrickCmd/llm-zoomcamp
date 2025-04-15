import unittest
from unittest.mock import patch, MagicMock
import json
import os

from assistant import (
    llm,
    evaluate_relevance,
    calculate_openai_cost,
    get_answer,
    indexer,
    es_strategy,
    ollama_client,
    openai_client
)
from llm_rag import LLMRAGHandler

# Mock environment variables
@patch.dict(os.environ, {"ELASTIC_URL": "http://test:9200", "OLLAMA_URL": "http://test:11434/v1/", "OPENAI_API_KEY": "test-api-key", "INDEX_NAME":"test_index", "MODEL_NAME":"test_model"})
class TestAssistantFunctions(unittest.TestCase):

    def setUp(self):
        # Create mock objects for external dependencies
        self.mock_indexer = MagicMock()
        self.mock_es_strategy = MagicMock()
        self.mock_ollama_client = MagicMock()
        self.mock_openai_client = MagicMock()

        #Patch the indexer and clients to use the mocked objects
        self.patcher_indexer = patch('assistant.indexer', self.mock_indexer)
        self.patcher_es_strategy = patch('assistant.es_strategy', self.mock_es_strategy)
        self.patcher_ollama_client = patch('assistant.ollama_client', self.mock_ollama_client)
        self.patcher_openai_client = patch('assistant.openai_client', self.mock_openai_client)

        self.patcher_indexer.start()
        self.patcher_es_strategy.start()
        self.patcher_ollama_client.start()
        self.patcher_openai_client.start()

    def tearDown(self):
        self.patcher_indexer.stop()
        self.patcher_es_strategy.stop()
        self.patcher_ollama_client.stop()
        self.patcher_openai_client.stop()

    def test_llm_ollama(self):
        # Mock the LLMRAGHandler's search method
        mock_rag_handler = MagicMock()
        mock_rag_handler.search.return_value = (
            "Ollama Answer",
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            0.5,
        )

        with patch("assistant.LLMRAGHandler", return_value=mock_rag_handler):
            answer, tokens, response_time = llm("Test query", "ollama/test-model")

        self.assertEqual(answer, "Ollama Answer")
        self.assertEqual(tokens, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        self.assertEqual(response_time, 0.5)
        mock_rag_handler.search.assert_called_once_with("Test query", k=5)

    def test_llm_openai(self):
        # Mock the LLMRAGHandler's search method
        mock_rag_handler = MagicMock()
        mock_rag_handler.search.return_value = (
            "OpenAI Answer",
            {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20},
            0.2,
        )

        with patch("assistant.LLMRAGHandler", return_value=mock_rag_handler):
            answer, tokens, response_time = llm("Test query", "openai/gpt-4o")

        self.assertEqual(answer, "OpenAI Answer")
        self.assertEqual(tokens, {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20})
        self.assertEqual(response_time, 0.2)
        mock_rag_handler.search.assert_called_once_with("Test query", k=5)

    def test_llm_unknown_model(self):
        with self.assertRaises(ValueError) as context:
            llm("Test query", "unknown/model")
        self.assertEqual(str(context.exception), "Unknown model choice: unknown/model")

    def test_evaluate_relevance_valid_json(self):
        # Mock the llm function to return a valid JSON string
        mock_llm_response = ('{"Relevance": "RELEVANT", "Explanation": "Good answer"}', {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, 0.5)
        with patch("assistant.llm", return_value=mock_llm_response) as mock_llm:
            relevance, explanation, tokens = evaluate_relevance("Test question", "Test answer")

        self.assertEqual(relevance, "RELEVANT")
        self.assertEqual(explanation, "Good answer")
        self.assertEqual(tokens,{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        mock_llm.assert_called_once_with(
            'You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.\nYour task is to analyze the relevance of the generated answer to the given question.\nBased on the relevance of the generated answer, you will classify it\nas "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".\n\nHere is the data for evaluation:\n\nQuestion: Test question\nGenerated Answer: Test answer\n\nPlease analyze the content and context of the generated answer in relation to the question\nand provide your evaluation in parsable JSON without using code blocks:\n\n{{\n  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",\n  "Explanation": "[Provide a brief explanation for your evaluation]"\n}}',
            "openai/gpt-4o-mini",
        )

    def test_evaluate_relevance_invalid_json(self):
        # Mock the llm function to return an invalid JSON string
        mock_llm_response = ("Invalid JSON", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, 0.5)
        with patch("assistant.llm", return_value=mock_llm_response):
            relevance, explanation, tokens = evaluate_relevance("Test question", "Test answer")

        self.assertEqual(relevance, "UNKNOWN")
        self.assertEqual(explanation, "Failed to parse evaluation")
        self.assertEqual(tokens,{"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})

    def test_calculate_openai_cost_gpt35(self):
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = calculate_openai_cost("openai/gpt-3.5-turbo", tokens)
        self.assertAlmostEqual(cost, 0.0025, places=4)

    def test_calculate_openai_cost_gpt4o(self):
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = calculate_openai_cost("openai/gpt-4o", tokens)
        self.assertAlmostEqual(cost, 0.06, places=4)

    def test_calculate_openai_cost_gpt4o_mini(self):
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = calculate_openai_cost("openai/gpt-4o-mini", tokens)
        self.assertAlmostEqual(cost, 0.06, places=4)

    def test_calculate_openai_cost_unknown_model(self):
        tokens = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        cost = calculate_openai_cost("openai/unknown", tokens)
        self.assertEqual(cost, 0)

    def test_get_answer(self):
        # Mock the llm and evaluate_relevance functions
        mock_llm_response = ("Test Answer", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, 0.5)
        mock_evaluate_response = ("RELEVANT", "Good answer", {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20})

        with patch("assistant.llm", return_value=mock_llm_response) as mock_llm, \
                patch("assistant.evaluate_relevance", return_value=mock_evaluate_response):
            result = get_answer("Test query", "openai/gpt-3.5-turbo")

        self.assertEqual(result["answer"], "Test Answer")
        self.assertEqual(result["response_time"], 0.5)
        self.assertEqual(result["relevance"], "RELEVANT")
        self.assertEqual(result["relevance_explanation"], "Good answer")
        self.assertEqual(result["model_used"], "openai/gpt-3.5-turbo")
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertEqual(result["completion_tokens"], 20)
        self.assertEqual(result["total_tokens"], 30)
        self.assertEqual(result["eval_prompt_tokens"], 5)
        self.assertEqual(result["eval_completion_tokens"], 15)
        self.assertEqual(result["eval_total_tokens"], 20)
        self.assertAlmostEqual(result["openai_cost"], 0.000045, places=6)

        mock_llm.assert_called_once_with("Test query", "openai/gpt-3.5-turbo", top_k=5)
    
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
