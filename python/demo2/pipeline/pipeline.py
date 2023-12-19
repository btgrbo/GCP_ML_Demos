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

from components import load_and_save_data
from components import training

PROJECT = "bt-int-ml-specialization"
REGION = "europe-west3"
PIPELINE_ROOT = f"gs://bt-int-ml-specialization-ml-demo2"
CURRENT_DIR = Path(__file__).parent


@dsl.pipeline(
    name='Vertex AI demo',
    description='Vertex AI demo'
)
def pipeline():
    """Pipeline to train a custom model on the Black Friday dataset."""

    # Load the dataset and split into train, validate, and test
    load_data_op = load_and_save_data(
        train_query='SELECT * FROM `bt-int-ml-specialization.demo2.vw-black-friday-train`',
        test_query='SELECT * FROM `bt-int-ml-specialization.demo2.vw-black-friday-test`',
    )

    training_op = training(
        train_file_parquet=load_data_op.outputs['train_data'],
        eval_file_parquet=load_data_op.outputs['test_data'],
#        model_file=f"{PIPELINE_ROOT}/mlmodels/model1.pkl",
#        eval_output_file_parquet=f"{PIPELINE_ROOT}/eval_output/eval_output.parquet",
#        train_script_path='/app/train.py',
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
    )

    aiplatform.init(project=PROJECT, location=REGION, experiment="demo2-black-friday-exp")

    run.submit(
        service_account=f"ml-demo2-executor@{PROJECT}.iam.gserviceaccount.com",
        experiment="demo2-black-friday-exp",
    )
    return run

if __name__ == "__main__":
    run_pipeline()
