import os
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform
from kfp import compiler
from kfp import dsl
from kfp.dsl import Metrics, Output

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "bt-int-ml-specialization")
REGION = "europe-west3"
PIPELINE_ROOT = f"gs://{PROJECT}-ml-demo1/pipelines"
CURRENT_DIR = Path(__file__).parent


@dsl.pipeline(name="demo1")
def make_pipeline(project_id: str):

    @dsl.component(base_image="python:3.11")
    def log_metric(metric_value: float, metrics: Output[Metrics]) -> str:
        text = f"My metric value is {metric_value}"
        metrics.log_metric("my_metric", metric_value)
        print(text)
        return text

    log_metric(metric_value=.7)


def run_pipeline():
    pipeline_file_path = str(CURRENT_DIR / "pipeline.json")
    compiler.Compiler().compile(make_pipeline, pipeline_file_path)

    aiplatform.init(
        project=PROJECT,
        location=REGION,
        experiment="exp-demo1",
    )

    now = datetime.now()
    job_id = f"demo1-{now:%Y-%m-%d-%H-%M-%S}"
    run = aiplatform.PipelineJob(
        project=PROJECT,
        location=REGION,
        display_name="demo1",
        template_path=pipeline_file_path,
        job_id=job_id,
        enable_caching=False,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT,
        },
    )

    run.submit(
        service_account=f"ml-demo1-executor@{PROJECT}.iam.gserviceaccount.com",
        experiment="exp-demo1",
    )
    return run


if __name__ == "__main__":
    run_pipeline()
