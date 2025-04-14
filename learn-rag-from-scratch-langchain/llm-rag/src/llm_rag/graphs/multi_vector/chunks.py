from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from llm_rag import llm
from llm_rag.indexing.multi_vector.chunks import retriever
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    search_results: list[Document]
    context: list[Document]
    answer: str


def retrieve(state: State):
    search_results = retriever.vectorstore.similarity_search(state["question"])
    retrieved_docs = retriever.invoke(state["question"])
    return {
        "search_results": search_results,
        "context": retrieved_docs,
    }


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
    agent_query = "What is task decomposition for LLM agents?"
    response = graph.invoke({"question": agent_query})
    rprint(Pretty(response, max_string=100, no_wrap=False))
    rprint(Markdown(response["answer"]))

    human_data_query = "What are main steps for collecting human data?"
    response = graph.invoke({"question": human_data_query})
    rprint(Pretty(response, max_string=100, no_wrap=False))
    rprint(Markdown(response["answer"]))
