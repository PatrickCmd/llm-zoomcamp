FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["streamlit", "run", "app.py"]