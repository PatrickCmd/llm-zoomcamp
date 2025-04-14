import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

from llm_rag import llm
from llm_rag.indexing.article import vectorstore

rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""


def get_unique_docs(documents: list[list[Document]]) -> list[Document]:
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def format_docs(docs: list[Document]) -> list[str]:
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    generated_questions: list[str]
    retrieved_docs: Annotated[list[list[Document]], operator.add]
    context: list[Document]
    answer: str


class RetrieverState(TypedDict):
    generated_question: str


def generate_queries(state: State, config: RunnableConfig):
    generated_questions_count = config["configurable"].get(
        "generated_questions_count", 5
    )
    include_original_question = config["configurable"].get(
        "include_original_question", True
    )

    questions = []
    query = state["question"]

    if include_original_question:
        questions.append(query)

    class MultiQueryGenerator(BaseModel):
        questions: list[str] = Field(
            ...,
            description="List of questions generated multiple perspectives based on user query",
            min_items=generated_questions_count,
            max_items=generated_questions_count,
        )

    structured_llm = llm.with_structured_output(
        MultiQueryGenerator, method="function_calling"
    )
    response = structured_llm.invoke(query)
    questions.extend(response.questions)

    return {"generated_questions": questions}


def assign_queries(state: State):
    return [
        Send("retrieve_docs", {"generated_question": question})
        for question in state["generated_questions"]
    ]


def retrieve_docs(state: RetrieverState):
    retrieved_docs = vectorstore.similarity_search(state["generated_question"])
    return {"retrieved_docs": [retrieved_docs]}


def aggregate_docs(state: State):
    retrieved_docs = state["retrieved_docs"]
    docs = get_unique_docs(retrieved_docs)
    return {"context": docs}


def generate_answer(state: State):
    docs_content = format_docs(state["context"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"], context=docs_content
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}


class ConfigSchema(BaseModel):
    generated_questions_count: int = Field(default=5, gt=1)
    include_original_question: bool = Field(default=True)


graph_builder = StateGraph(State, ConfigSchema)

graph_builder.add_node("generate_queries", generate_queries)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("aggregate_docs", aggregate_docs)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_queries")
graph_builder.add_conditional_edges(
    "generate_queries", assign_queries, ["retrieve_docs"]
)
graph_builder.add_edge("retrieve_docs", "aggregate_docs")
graph_builder.add_edge("aggregate_docs", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What is task decomposition for LLM agents?"
    config = {
        "configurable": {
            "generated_questions_count": 3,
            "include_original_question": False,
        }
    }
    response = graph.invoke({"question": query}, config=config)

    rprint(Pretty(response, max_depth=2))
    rprint(Markdown(response["answer"]))
