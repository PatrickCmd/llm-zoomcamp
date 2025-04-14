import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from llm_rag import embeddings


def load_documents():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs


def prepare_vectorstore(docs, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents=splits)
    return vectorstore


docs = load_documents()
vectorstore = prepare_vectorstore(docs, embeddings)
retriever = vectorstore.as_retriever()
