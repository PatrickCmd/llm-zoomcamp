import uuid

import bs4
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.stores import InMemoryByteStore
from langchain_core.vectorstores import InMemoryVectorStore
from llm_rag import embeddings

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


def prepare_retriever(docs, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore(embeddings)
    store = InMemoryByteStore()
    id_key = "split_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )

    split_ids = [str(uuid.uuid4()) for _ in splits]

    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )

    all_sub_splits = []

    for i, split in enumerate(splits):
        split_id = split_ids[i]
        sub_splits = child_text_splitter.split_documents([split])

        for sub_split in sub_splits:
            sub_split.metadata[id_key] = split_id

        all_sub_splits.extend(sub_splits)

    retriever.vectorstore.add_documents(all_sub_splits)
    retriever.docstore.mset(list(zip(split_ids, splits)))

    return retriever


docs = load_documents()
retriever = prepare_retriever(docs, embeddings)
