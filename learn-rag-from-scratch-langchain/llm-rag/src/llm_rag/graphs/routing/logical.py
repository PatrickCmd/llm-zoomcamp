from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from llm_rag import llm
from pydantic import BaseModel, Field
from rich import print as rprint

system_prompt = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""


class RouteInfo(BaseModel):
    """Route a user query to the most relevant data source."""

    data_source: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which data source would be most relevant for answering their question",
    )


structured_llm = llm.with_structured_output(RouteInfo, method="function_calling")


class State(TypedDict):
    question: str
    data_source: str
    context: str
    answer: str


def select_data_source(state: State):
    response = structured_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"data_source": response.data_source}


def route_query(state: State) -> Literal["python_docs", "js_docs", "golang_docs"]:
    return state["data_source"]


def retrieve_python_docs(state: State):
    return {"context": "Python documentation"}


def retrieve_js_docs(state: State):
    return {"context": "Javascript documentation"}


def retrieve_golang_docs(state: State):
    return {"context": "Go documentation"}


def generate_answer(state: State):
    return {"answer": f"Answer based on {state['context']}"}


graph_builder = StateGraph(State)

graph_builder.add_node("select_data_source", select_data_source)
graph_builder.add_node("python_docs", retrieve_python_docs)
graph_builder.add_node("js_docs", retrieve_js_docs)
graph_builder.add_node("golang_docs", retrieve_golang_docs)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "select_data_source")
graph_builder.add_conditional_edges(
    "select_data_source", route_query, ["python_docs", "js_docs", "golang_docs"]
)
graph_builder.add_edge("python_docs", "generate_answer")
graph_builder.add_edge("js_docs", "generate_answer")
graph_builder.add_edge("golang_docs", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    python_query = """Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """
    response = graph.invoke({"question": python_query})
    rprint(response)

    javascript_query = """Which arguments has getElementById function?"""
    response = graph.invoke({"question": javascript_query})
    rprint(response)

    golang_query = """What is struct?"""
    response = graph.invoke({"question": golang_query})
    rprint(response)
