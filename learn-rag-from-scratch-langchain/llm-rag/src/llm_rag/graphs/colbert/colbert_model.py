from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from llm_rag import llm
from llm_rag.indexing.colbert_model import retriever
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = format_docs(state["context"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"], context=docs_content
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}


graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    queries = [
        "What is task decomposition for LLM agents?",
        "What are main steps for collecting human data?",
    ]

    for query in queries:
        response = graph.invoke({"question": query})
        rprint(Pretty(response, no_wrap=False))
        rprint(Markdown(response["answer"]))
