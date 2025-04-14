from typing import TypedDict

from langchain_community.utils.math import cosine_similarity
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from llm_rag import embeddings, llm
from rich import print as rprint
from rich.markdown import Markdown

prompt_names = ["PHYSICS", "MATH", "OTHER"]


physics_prompt_template = """You are a very smart physics professor.
You are great at answering questions about physics in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{question}"""


math_prompt_template = """You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{question}"""


other_prompt_template = f"""You are a helpful assistant. You are great at answering all questions not from the following themes: {prompt_names[:-1]}

Here is a question:
{{question}}"""


prompt_templates = [
    physics_prompt_template,
    math_prompt_template,
    other_prompt_template,
]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


class State(TypedDict):
    question: str
    most_similar_prompt_idx: int
    most_similar_prompt_name: str
    answer: str


def select_route_prompt(state: State):
    query_embedding = embeddings.embed_query(state["question"])
    query_similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar_prompt_idx = query_similarity.argmax()
    return {
        "most_similar_prompt_idx": most_similar_prompt_idx,
        "most_similar_prompt_name": prompt_names[most_similar_prompt_idx],
    }


def generate_answer(state: State):
    route_prompt = prompt_templates[state["most_similar_prompt_idx"]].format(
        question=state["question"]
    )
    response = llm.invoke([HumanMessage(content=route_prompt)])
    return {"answer": response.content}


graph_builder = StateGraph(State)

graph_builder.add_node("select_route_prompt", select_route_prompt)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "select_route_prompt")
graph_builder.add_edge("select_route_prompt", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    queries = [
        "What's a black hole",
        "What is the square root of 81",
        "Hello! How are you?",
    ]

    for query in queries:
        print(query)
        response = graph.invoke({"question": query})
        rprint(response)
        rprint(Markdown(response["answer"]))
        rprint("=" * 50)
