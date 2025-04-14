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

summarization_prompt_template = "Summarize the following document:\n\n{doc}"


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


@chain
def summarize_document(doc):
    summarization_prompt = summarization_prompt_template.format(doc=doc.page_content)
    response = llm.invoke([HumanMessage(content=summarization_prompt)])
    return response.content


def prepare_retriever(docs, embeddings):
    vectorstore = InMemoryVectorStore(embeddings)
    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    summaries = summarize_document.batch(docs, {"max_concurrency": len(docs)})
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever


docs = load_documents()
retriever = prepare_retriever(docs, embeddings)
