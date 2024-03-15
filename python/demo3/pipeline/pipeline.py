# pyright: reportCallIssue=false

from datetime import datetime
from pathlib import Path

from components.check_endpoint import check_endpoint
from google.cloud import aiplatform
from google_cloud_pipeline_components._implementation import model as ai_model_exp
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from kfp import compiler, dsl

PROJECT = "bt-int-ml-specialization"
REGION = "europe-west4"
PIPELINE_ROOT = "gs://bt-int-ml-specialization-ml-demo3"
CURRENT_DIR = Path(__file__).parent


@dsl.pipeline(name="Vertex AI demo", description="Vertex AI demo")
def pipeline(endpoint_id: str, model_name: str, model_version: int):
    """Pipeline deploy a new MRI tumor detection model version"""

    endpoint_name = f"projects/{PROJECT}/locations/{REGION}/endpoints/{endpoint_id}"
    endpoint_uri = f"https://{REGION}-aiplatform.googleapis.com/v1/{endpoint_name}"

    endpoint_importer = dsl.importer(
        artifact_uri=endpoint_uri,
        artifact_class=artifact_types.VertexEndpoint,
        metadata={"resourceName": endpoint_name},
    )

    model = ai_model_exp.GetVertexModelOp(model_name=model_name, model_version=str(model_version))

    deploy_op = ModelDeployOp(
        model=model.outputs["model"],
        endpoint=endpoint_importer.output,
        automatic_resources_min_replica_count=1,
        automatic_resources_max_replica_count=1,
    )

    check_endpoint(project=PROJECT, endpoint_name=endpoint_name).after(deploy_op)


def run_pipeline():
    pipeline_file_path = str(CURRENT_DIR / "pipeline.json")
    compiler.Compiler().compile(pipeline, pipeline_file_path)  # type: ignore

    now = datetime.now()
    job_id = f"demo3-{now:%Y-%m-%d-%H-%M-%S}"
    run = aiplatform.PipelineJob(
        project=PROJECT,
        location=REGION,
        display_name="demo3",
        template_path=pipeline_file_path,
        job_id=job_id,
        enable_caching=True,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "endpoint_id": "8009005424441294848",
            "model_name": "projects/bt-int-ml-specialization/locations/europe-west4/models/8959176984886247424",
            "model_version": 1,
        },
    )

    aiplatform.init(project=PROJECT, location=REGION)

    run.submit(
        service_account=f"ml-demo2-executor@{PROJECT}.iam.gserviceaccount.com",
    )
    return run


if __name__ == "__main__":
    run_pipeline()
