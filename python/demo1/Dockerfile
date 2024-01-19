FROM python:3.10-slim

WORKDIR /app

RUN ls -l

COPY requirements.txt python/demo1/requirements.txt
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

#COPY test_data/eval_parquet_file.parquet eval_parquet_file.parquet
#COPY test_data/train_parquet_file.parquet train_parquet_file.parquet

COPY main.py main.py

CMD ["python", "main.py", "--help"]