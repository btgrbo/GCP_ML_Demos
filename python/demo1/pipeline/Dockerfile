FROM python:3.10-slim

WORKDIR /app

COPY python/demo1/requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY python/demo1/pipeline/pipeline.py pipeline.py

CMD ["python", "pipeline.py", "--help"]