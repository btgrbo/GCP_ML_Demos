FROM python:3.10-slim

WORKDIR / app

COPY pipfile.LOCK pipfile.LOCK
RUN pip install -r pipfile.LOCK

COPY main.py main.py
COPY training training

CMD [ "python", "main.py", "--help" ]
