import uuid

import bs4
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain_core.stores import InMemoryByteStore
from langchain_core.vectorstores import InMemoryVectorStore
from llm_rag import embeddings, llm
from pydantic import BaseModel, Field

hypothetical_questions_prompt_template = "Generate a list of exactly {hypothetical_questions_count} hypothetical questions that the below document could be used to answer:\n\n{doc}"


def load_documents():
    articles = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    ]
    loader = WebBaseLoader(
        web_paths=articles,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs


class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions."""

    questions: list[str] = Field(..., description="List of questions")


@chain
def generate_hypothetical_questions(doc, hypothetical_questions_count=3):
    hypothetical_questions_prompt = hypothetical_questions_prompt_template.format(
        hypothetical_questions_count=hypothetical_questions_count, doc=doc.page_content
    )
    structured_llm = llm.with_structured_output(HypotheticalQuestions)
    response = structured_llm.invoke(
        [HumanMessage(content=hypothetical_questions_prompt)]
    )
    return response.questions


def prepare_retriever(docs, embeddings):
    vectorstore = InMemoryVectorStore(embeddings)
    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    hypothetical_questions = generate_hypothetical_questions.batch(
        docs, {"max_concurrency": len(docs)}
    )
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    question_docs = []

    for i, questions in enumerate(hypothetical_questions):
        question_docs.extend(
            [
                Document(page_content=question, metadata={id_key: doc_ids[i]})
                for question in questions
            ]
        )

    retriever.vectorstore.add_documents(question_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever


docs = load_documents()
retriever = prepare_retriever(docs, embeddings)
