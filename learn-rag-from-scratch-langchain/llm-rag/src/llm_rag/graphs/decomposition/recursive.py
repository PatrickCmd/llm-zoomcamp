from typing import Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown

from llm_rag import llm
from llm_rag.indexing.article import vectorstore

decomposition_prompt_template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered sequentially.
Generate multiple search queries related to: {question}"""


recursive_prompt_template = """Here is the question you need to answer:
<question>
{question}
</question>

Here are any available background question + answer pairs:
<question_answer_pairs>
{qa_pairs}
</question_answer_pairs>

Here is additional context relevant to the question: 
<context>
{context}
</context>

Use the above context and any background question + answer pairs to answer the question:
<question>
{question}
</question>
"""


def format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}  \nAnswer:\n{answer}\n\n"


class State(TypedDict):
    question: str
    all_questions: list[str]
    current_question_idx: int
    qa_pairs: list[str]
    context: list[Document]
    answer: str


def generate_sub_questions(state: State, config: RunnableConfig) -> list[str]:
    max_generated_sub_questions_count = config["configurable"].get(
        "max_generated_sub_questions_count", 3
    )
    query = state["question"]

    class SubQuestionsGenerator(BaseModel):
        sub_questions: list[str] = Field(
            ...,
            description="List of generated sub-problems / sub-questions",
            max_items=max_generated_sub_questions_count,
        )

    structured_llm = llm.with_structured_output(
        SubQuestionsGenerator, method="function_calling"
    )
    decomposition_prompt = decomposition_prompt_template.format(question=query)
    response = structured_llm.invoke([HumanMessage(content=decomposition_prompt)])
    questions = response.sub_questions + [query]

    return {"all_questions": questions, "current_question_idx": 0}


def retrieve_docs(state: State):
    question = state["all_questions"][state["current_question_idx"]]
    retrieved_docs = vectorstore.similarity_search(question)
    return {"context": retrieved_docs}


def generate_answer(state: State):
    question = state["all_questions"][state["current_question_idx"]]
    recursive_prompt = recursive_prompt_template.format(
        question=question, qa_pairs=state.get("qa_pairs", ""), context=state["context"]
    )
    answer = llm.invoke([HumanMessage(content=recursive_prompt)])
    qa_pair = format_qa_pair(question, answer.content)
    qa_pairs = state.get("qa_pairs", "") + qa_pair

    if state["current_question_idx"] == len(state["all_questions"]) - 1:
        return {"answer": answer.content}
    else:
        return {
            "qa_pairs": qa_pairs,
            "current_question_idx": state["current_question_idx"] + 1,
        }


def check_answer_status(state: State) -> Literal["Next sub-question", "Final answer"]:
    if state.get("answer"):
        return "Final answer"
    else:
        return "Next sub-question"


class ConfigSchema(BaseModel):
    max_generated_sub_questions_count: int = Field(default=3, gt=1)


graph_builder = StateGraph(State, ConfigSchema)

graph_builder.add_node("generate_sub_questions", generate_sub_questions)
graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_sub_questions")
graph_builder.add_edge("generate_sub_questions", "retrieve_docs")
graph_builder.add_edge("retrieve_docs", "generate_answer")
graph_builder.add_conditional_edges(
    "generate_answer",
    check_answer_status,
    {"Next sub-question": "retrieve_docs", "Final answer": END},
)

graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What are the main components of an LLM-powered autonomous agent system?"
    config = {
        "configurable": {
            "max_generated_sub_questions_count": 3,
        }
    }

    for stream_mode, event in graph.stream(
        {"question": query},
        stream_mode=["messages", "updates"],
        config=config,
    ):
        match stream_mode:
            case "messages":
                message, metadata = event
                print(message.content, end="", flush=True)
            case "updates":
                rprint(event)

    rprint(Markdown(event["generate_answer"]["answer"]))
