from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from llm_rag import llm
from llm_rag.indexing.article import vectorstore
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

step_back_prompt_template = "You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
step_back_prompt = ChatPromptTemplate.from_messages(
    [("system", step_back_prompt_template), few_shot_prompt, ("human", "{question}")]
)


final_answer_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{context}
{step_back_context}

Original Question: {question}
Answer:"""


class State(TypedDict):
    question: str
    context: list[Document]
    step_back_question: str
    step_back_context: list[Document]
    answer: str


def retrieve_docs(state: State):
    question = state["question"]
    retrieved_docs = vectorstore.similarity_search(question)
    return {"context": retrieved_docs}


def generate_step_back_question(state: State):
    step_back_prompt_messages = step_back_prompt.format(question=state["question"])
    step_back_question = llm.invoke(step_back_prompt_messages)
    return {"step_back_question": step_back_question.content}


def retrieve_step_back_docs(state: State):
    step_back_question = state["step_back_question"]
    retrieved_step_back_docs = vectorstore.similarity_search(step_back_question)
    return {"step_back_context": retrieved_step_back_docs}


def generate_answer(state: State):
    final_answer_prompt = final_answer_prompt_template.format(
        context=state["context"],
        step_back_context=state["step_back_context"],
        question=state["question"],
    )
    response = llm.invoke([HumanMessage(content=final_answer_prompt)])
    return {"answer": response.content}


graph_builder = StateGraph(State)

graph_builder.add_node("retrieve_docs", retrieve_docs)
graph_builder.add_node("generate_step_back_question", generate_step_back_question)
graph_builder.add_node("retrieve_step_back_docs", retrieve_step_back_docs)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "retrieve_docs")
graph_builder.add_edge("retrieve_docs", "generate_step_back_question")
graph_builder.add_edge("generate_step_back_question", "retrieve_step_back_docs")
graph_builder.add_edge("retrieve_step_back_docs", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What is task decomposition for LLM agents?"
    response = graph.invoke(
        {"question": query},
    )

    rprint(Pretty(response, max_depth=2))
    rprint(Markdown(response["answer"]))
