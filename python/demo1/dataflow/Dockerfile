FROM gcr.io/dataflow-templates-base/python310-template-launcher-base:latest

WORKDIR /dataflow/template

# Do not include `apache-beam` in requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir -U -r ./requirements.txt && \
    pip install --no-cache-dir apache-beam[gcp]

COPY setup.py .
COPY src ./src
COPY main_inference.py ./main.py

ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE="/dataflow/template/requirements.txt"
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/dataflow/template/main.py"
ENV FLEX_TEMPLATE_PYTHON_SETUP_FILE="/dataflow/template/setup.py"