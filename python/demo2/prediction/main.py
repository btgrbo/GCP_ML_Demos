import os
import pickle
from io import BytesIO
from typing import Any, Protocol

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel


class Request(BaseModel):
    instances: list[dict[str, Any]]


class Response(BaseModel):
    predictions: list[float]


class Model(Protocol):

    def predict(self, Lx: pd.DataFrame) -> np.ndarray: ...


def load_model(gcs_uri: str, project_id: str) -> Model:
    client = storage.Client(project=project_id)
    tmp_model = BytesIO()
    client.download_blob_to_file(gcs_uri, tmp_model)
    tmp_model.seek(0)
    return pickle.load(tmp_model)


if __name__ == "__main__":
    app = FastAPI()

    model_artifact_location = os.environ["MODEL_URI"]

    print(f"{model_artifact_location=}", flush=True)
    if not model_artifact_location:
        raise RuntimeError("MODEL_URI is not set!")

    model = load_model(
        gcs_uri=f"{model_artifact_location}",
        project_id="bt-int-ml-specialization",
    )

    @app.get("/ping")
    async def healthy():
        return {"Healthy"}

    @app.post("/predictions")
    async def predict(request: Request) -> Response:
        df = pd.DataFrame.from_records(request.instances)
        return Response(predictions=model.predict(df).tolist())

    uvicorn.run(app, host="0.0.0.0", port=8080)
