# Module 4: Evaluation and Monitoring

See course module 4 content [here](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/04-monitoring)

In this module, we learn how to evaluate and monitor our LLM and RAG system.

In the evaluation part, we assess the quality of our entire RAG system before it goes live.

In the monitoring part, we collect, store and visualize metrics to assess the answer quality of a deployed LLM. We also collect chat history and user feedback.

## Offline vs Online (RAG) evaluation

### Generating data for offline RAG evaluation

Links:

- [notebook](./offline-rag-evaluation.ipynb)
- [results-llama3_8b_8192.csv](./data/results-llama3_8b_8192.csv) (answers from llama3-8b-8192)
- [results-gemma_7b_it.csv](./data/results-gemma_7b_it.csv) (answers from gemma-7b-it)

### Offline RAG evaluation: cosine similarity

Content

- A->Q->A' cosine similarity
- Evaluating llama3-8b-8192
- Evaluating gemma-7b-it
- Evaluating gemma2-9b-it

### Offline RAG evaluation: LLM as a judge

- LLM as a judge
- A->Q->A' evaluation
- Q->A evaluation

Links:

- [notebook](./offline-rag-evaluation.ipynb)
- [evaluations-aqa.csv](./data/evaluations-aqa.csv) (A->Q->A evaluation results)
- [evaluations-qa.csv](./data/evaluations-qa.csv) (Q->A evaluation results)

### Capturing user feedback

> You can see the prompts and the output from claude [here](./code.md)

Content

- Adding +1 and -1 buttons
- Setting up a postgres database
- Putting everything in docker compose

```sh
pip install pgcli
pgcli -h localhost -U your_username -d course_assistant -W
```

Links:

- [final code](./app/)
- [intermediate code from claude](https://github.com/PatrickCmd/llm-zoomcamp/blob/main/04-Monitoring/code.md#46-capturing-user-feedback)

### Capturing user feedback: part 2

- adding vector search
- adding OpenAI

Links:

- [final code](./app/)
- [intermediate code from claude](https://github.com/PatrickCmd/llm-zoomcamp/blob/main/04-Monitoring/code.md#462-capturing-user-feedback-part-2)


### Monitoring the system

- Setting up Grafana
- Tokens and costs
- QA relevance
- User feedback
- Other metrics

Links:

- [final code](./app/)
- [SQL queries for Grafana](./grafana.md)
- [intermediate code from claude](https://github.com/PatrickCmd/llm-zoomcamp/blob/main/04-Monitoring/code.md#47-monitoring)


### Extra Grafana video

- Grafana variables
- Exporting and importing dashboards

Links:

- [SQL queries for Grafana](./grafana.md)
- [Grafana dashboard](./app/graphana_dashboard.png)
