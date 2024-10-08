# Module 6: Best practices

In this module, we'll cover the techniques that could improve your RAG pipeline.

## 6.1 Techniques to Improve RAG Pipeline

* Small-to-Big chunk retrieval
* Leveraging document metadata
* Hybrid search
* User query rewriting
* Document reranking

Links:
* [Slides](llm-zoomcamp-best-practicies.pdf)
* [Five Techniques for Improving RAG Chatbots - Nikita Kozodoi [Video]](https://www.youtube.com/watch?v=xPYmClWk5O8)
* [Survey on RAG techniques [Article]](https://arxiv.org/abs/2312.10997)

## 6.2 Hybrid search

* Hybrid search strategy
* Hybrid search in Elasticsearch

Links:
* [Notebook](hybrid-search-and-reranking-es.ipynb)
* [Hybrid search [Elasticsearch Guide]](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#_combine_approximate_knn_with_other_features)
* [Hybrid search [Tutorial]](https://www.elastic.co/search-labs/tutorials/search-tutorial/vector-search/hybrid-search)

## 6.3 Document Reranking

* Reranking concept and metrics
* Reciprocal Rank Fusion (RRF)
* Handmade raranking implementation

Links:
* [Reciprocal Rank Fusion (RRF) method [Elasticsearch Guide]](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
* [RRF method [Article]](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
* [Elasticsearch subscription plans](https://www.elastic.co/subscriptions)

We should pull and run a docker container with Elasticsearch 8.9.0 or higher in order to use reranking based on RRF algorithm:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.9.0
```

## 6.4 Hybrid search with LangChain

* LangChain: Introduction
* ElasticsearchRetriever
* Hybrid search implementation

```bash
pip install -qU langchain langchain-elasticsearch langchain-huggingface
```

Links:
* [Notebook](hybrid-search-langchain.ipynb)
* [Chatbot Implementation [Tutorial]](https://www.elastic.co/search-labs/tutorials/chatbot-tutorial/implementation)
* [ElasticsearchRetriever](https://python.langchain.com/v0.2/docs/integrations/retrievers/elasticsearch_retriever/)