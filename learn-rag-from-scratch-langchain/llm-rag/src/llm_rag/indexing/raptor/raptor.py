from bs4 import BeautifulSoup as Soup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from llm_rag import embeddings
from llm_rag.indexing.raptor.utils import recursive_embed_cluster_summarize


def load_documents():
    url = "https://langchain-ai.github.io/langgraph/tutorials/introduction/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    introduction_docs = loader.load()

    url = "https://langchain-ai.github.io/langgraph/concepts/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    concepts_docs = loader.load()

    docs = introduction_docs + concepts_docs
    return docs


def prepare_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    leaf_texts = [doc.page_content for doc in splits]
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)

    all_docs = [
        Document(page_content=text, metadata={"level": 0}) for text in leaf_texts
    ]

    for level in sorted(results.keys()):
        all_docs.extend(
            [
                Document(page_content=summary, metadata={"level": level})
                for summary in results[level][1]["summaries"]
            ]
        )

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents=all_docs)
    return vectorstore


docs = load_documents()
vectorstore = prepare_vectorstore(docs)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10},
)
