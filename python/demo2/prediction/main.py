import os
import pickle
import re
from typing import Any, Protocol

import pandas as pd
import uvicorn
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel


class Request(BaseModel):
    instances: list[dict[str, Any]]


class Response(BaseModel):
    scores: list[float]


class Model(Protocol):
    def predict(x: pd.DataFrame) -> list[float]: ...


def load_model(gcs_uri: str, project_id: str) -> Model:
    client = storage.Client(project=project_id)
    pattern = r"gs://([^/]+)/(.+)"
    bucket_, file_path = re.match(pattern, gcs_uri).groups()
    bucket = client.get_bucket(bucket_)
    blob = bucket.get_blob(file_path)
    model_bytes = blob.download_as_bytes()
    model = pickle.loads(model_bytes)
    return model


if __name__ == "__main__":
    app = FastAPI()

    model = load_model(
        gcs_uri=os.environ["AIP_STORAGE_URI"],
        project_id="bt-int-ml-specialization",
    )

    @app.get("/ping")
    async def healthy():
        return {"Healthy"}

    @app.post("/predictions", response_model=Response)
    async def predict(request: Request):
        df = pd.DataFrame.from_records(request.instances)
        return model.predict(df)

    uvicorn.run(app, host="0.0.0.0", port=8080)
