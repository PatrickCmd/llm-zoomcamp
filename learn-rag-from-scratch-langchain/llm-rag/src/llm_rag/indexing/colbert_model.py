import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from ragatouille import RAGPretrainedModel


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


def prepare_model(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    docs_texts = [doc.page_content for doc in splits]
    docs_metadatas = [doc.metadata for doc in splits]

    model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    model.index(
        collection=docs_texts,
        document_metadatas=docs_metadatas,
        index_name="blog",
        split_documents=False,
    )

    return model


docs = load_documents()
model = prepare_model(docs)
retriever = model.as_langchain_retriever(k=10)
