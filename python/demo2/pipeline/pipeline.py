"""
Vertex AI pipeline definition with kubeflow pipelines DSL.

This pipeline
  - loads the train and test data set for demo 2 (black friday)
  - trains a custom model on the train dataset
  - executes the trained model
"""

from datetime import datetime
from pathlib import Path

import components
from google.cloud import aiplatform
from google_cloud_pipeline_components._implementation import model as ai_model_exp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from kfp import compiler, dsl

PROJECT = "bt-int-ml-specialization"
REGION = "europe-west3"
PIPELINE_ROOT = "gs://bt-int-ml-specialization-ml-demo2"
CURRENT_DIR = Path(__file__).parent


@dsl.pipeline(
    name='Vertex AI demo',
    description='Vertex AI demo'
)
def pipeline(display_name: str, data_dir: str, eval_split_ratio: float):
    """Pipeline to train a custom model on the Black Friday dataset."""

    data = dsl.importer(
        artifact_uri=data_dir,
        artifact_class=dsl.Dataset,
    )

    preprocessing_op = components.transform(
        data=data.outputs["artifact"]
    )

    datasplit_op = components.train_eval_split(
        data=preprocessing_op.outputs["data_proc"],
        eval_ratio=eval_split_ratio,
    )

    training_op = components.training(
        train_file_parquet=datasplit_op.outputs["data_train"],
        eval_file_parquet=datasplit_op.outputs["data_test"],
    )

    predictor_pipeline_op = components.create_pipeline(
        preprocessor=preprocessing_op.outputs["preprocessor"],
        model=training_op.outputs["model"],
    )

    unmanaged_model_importer = components.import_model(
        model_artifact=predictor_pipeline_op.outputs["pipeline"],
    )

    parent_model_name = "projects/bt-int-ml-specialization/locations/europe-west3/models/6679339624692711424"
    parent_model = ai_model_exp.GetVertexModelOp(model_name=parent_model_name)

    model_upload_op = ModelUploadOp(
        display_name=display_name,
        unmanaged_container_model=unmanaged_model_importer.outputs["artifact"],
        project=PROJECT,
        location=REGION,
        parent_model=parent_model.outputs["model"],
    )

    endpoint = EndpointCreateOp(
        display_name=f"{display_name}_20240222_endpoint",
        location=REGION,
    )

    _ = ModelDeployOp(
        model=model_upload_op.outputs["model"],
        endpoint=endpoint.outputs["endpoint"],
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        service_account=f"ml-demo2-predictor@{PROJECT}.iam.gserviceaccount.com",
    )


def run_pipeline():
    pipeline_file_path = str(CURRENT_DIR / "pipeline.json")
    compiler.Compiler().compile(pipeline, pipeline_file_path)

    now = datetime.now()
    job_id = f"demo2-{now:%Y-%m-%d-%H-%M-%S}"
    run = aiplatform.PipelineJob(
        project=PROJECT,
        location=REGION,
        display_name="demo2",
        template_path=pipeline_file_path,
        job_id=job_id,
        enable_caching=True,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "display_name": "demo2",
            "data_dir": "gs://bt-int-ml-specialization-ml-demo2/training_data/train_20240207.parquet",
            "eval_split_ratio": 0.2,
        },
    )

    aiplatform.init(project=PROJECT, location=REGION,
                    experiment="demo2-black-friday-exp")

    run.submit(
        service_account=f"ml-demo2-executor@{PROJECT}.iam.gserviceaccount.com",
        experiment="demo2-black-friday-exp",
    )
    return run


if __name__ == "__main__":
    run_pipeline()
