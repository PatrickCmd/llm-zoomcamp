from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from llm_rag.indexing.self_query import retriever
from rich import print as rprint
from rich.pretty import Pretty


class State(TypedDict):
    question: str
    context: list[Document]


def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "Which videos are 7 to 10 minutes long",
        "Videos published in March 2024",
        "Find tutorials with views not less than 100k",
        "Which videos should I watch on the topic of routing",
        "Which 1 video should I watch on the topic of routing",
    ]

    for question in questions:
        print(question)
        response = graph.invoke({"question": question})
        rprint(Pretty(response, max_string=100, no_wrap=False))
