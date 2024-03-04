from google_cloud_pipeline_components.types import artifact_types
from kfp import dsl

components = ["pyarrow==15.0.0", "pandas==2.2.1", "google-cloud-aiplatform", "google-cloud-pipeline-components"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def evaluate_model(
    project_id: str,
    data: dsl.Input[dsl.Dataset],
    endpoint: dsl.Input[artifact_types.VertexEndpoint],
    predictions: dsl.Output[dsl.Dataset],
):

    import pandas as pd
    from google.cloud import aiplatform

    ep = aiplatform.Endpoint(
        endpoint_name=endpoint.metadata["resourceName"], location="europe-west3", project=project_id
    )

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(data.path)

    chunks = []
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=128)):
        df = batch.to_pandas()
        targets = df[["Purchase"]]
        features = df.drop(columns=["Purchase"]).to_dict(orient="records")
        targets["Prediction"] = ep.predict(instances=features).predictions
        chunks.append(targets)
        print(f"batch {i} done.")

    df_out = pd.concat(chunks)

    df_out.to_parquet(predictions.path)


# if __name__ == "__main__":
#     from unittest.mock import MagicMock, Mock

#     endpoint = Mock()
#     endpoint.name = "projects/738673379845/locations/europe-west3/endpoints/7994162017466318848"
#     data = MagicMock(path="~/Downloads/training_data-train_20240207.parquet")
#     predictions = MagicMock(path="/tmp/predictions.parquet")

#     evaluate_model.python_func("bt-int-ml-specialization", data, endpoint, predictions)
