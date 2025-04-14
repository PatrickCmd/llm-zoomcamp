from typing import TypedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

from llm_rag import embeddings, llm
from llm_rag.indexing.article import vectorstore

hyde_prompt_template = """Please write a passage to answer the question
Question: {question}
Passage:"""


rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    generated_documents: list[str]
    hyde_embeddings: np.ndarray
    context: list[Document]
    answer: str


def generate_documents(state: State, config: RunnableConfig) -> list[Document]:
    generated_documents_count = config["configurable"].get(
        "generated_documents_count", 3
    )

    hyde_prompt = hyde_prompt_template.format(question=state["question"])
    generated_documents = llm.batch([hyde_prompt] * generated_documents_count)

    return {
        "generated_documents": [document.content for document in generated_documents]
    }


def calculate_hyde_embeddings(state: State):
    question_embeddings = np.array(embeddings.embed_query(state["question"]))
    generated_documents_embeddings = np.array(
        embeddings.embed_documents(state["generated_documents"])
    )
    hyde_embeddings = np.vstack(
        [question_embeddings, generated_documents_embeddings]
    ).mean(axis=0)
    return {"hyde_embeddings": list(hyde_embeddings)}


def get_relevant_documents(state: State):
    documents = vectorstore.similarity_search_by_vector(state["hyde_embeddings"])
    return {"context": documents}


def generate_answer(state: State):
    docs_content = format_docs(state["context"])
    rag_prompt = rag_prompt_template.format(
        context=docs_content, question=state["question"]
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}


class ConfigSchema(BaseModel):
    generated_documents_count: int = Field(default=3, gt=0)


graph_builder = StateGraph(State, ConfigSchema)

graph_builder.add_node("generate_documents", generate_documents)
graph_builder.add_node("calculate_hyde_embeddings", calculate_hyde_embeddings)
graph_builder.add_node("get_relevant_documents", get_relevant_documents)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_documents")
graph_builder.add_edge("generate_documents", "calculate_hyde_embeddings")
graph_builder.add_edge("calculate_hyde_embeddings", "get_relevant_documents")
graph_builder.add_edge("get_relevant_documents", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What is task decomposition for LLM agents?"
    config = {
        "configurable": {
            "generated_documents_count": 5,
        }
    }
    response = graph.invoke(
        {"question": query},
        config=config,
    )

    rprint(Pretty(response, max_depth=2, max_length=20))
    rprint(Markdown(response["answer"]))
