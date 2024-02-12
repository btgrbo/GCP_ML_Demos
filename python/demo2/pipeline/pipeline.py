"""
Vertex AI pipeline definition with kubeflow pipelines DSL.

This pipeline
  - loads the train and test data set for demo 2 (black friday)
  - trains a custom model on the train dataset
  - executes the trained model
"""

from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform
from kfp import dsl, compiler

import components

PROJECT = "bt-int-ml-specialization"
REGION = "europe-west3"
PIPELINE_ROOT = f"gs://bt-int-ml-specialization-ml-demo2"
CURRENT_DIR = Path(__file__).parent


@dsl.pipeline(
    name='Vertex AI demo',
    description='Vertex AI demo'
)
def pipeline(data_dir: str, test_split_ratio: float):
    """Pipeline to train a custom model on the Black Friday dataset."""

    data = dsl.importer(
        artifact_uri=data_dir,
        artifact_class=dsl.Dataset,
    )

    preprocessing_op = components.transform(
        data=data.outputs["artifact"]
    )

    datasplit_op = components.train_test_split(
        data=preprocessing_op.outputs["data_proc"],
        test_ratio=test_split_ratio,
    )

    # training_op = components.training(
    #     train_file_parquet=datasplit_op.outputs['data_train'],
    #     eval_file_parquet=datasplit_op.outputs['data_test'],
    # )


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
            "data_dir": "gs://bt-int-ml-specialization-ml-demo2/training_data/train_20240207.parquet",
            "test_split_ratio": 0.2,
        }
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
