# pyright: reportCallIssue=false

"""
Vertex AI pipeline definition with kubeflow pipelines DSL.

"""

from datetime import datetime
from pathlib import Path

import components
from google.cloud import aiplatform
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.dataflow import DataflowFlexTemplateJobOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.model_evaluation import (
    ModelEvaluationRegressionOp,
)
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp
from kfp import compiler, dsl

now = datetime.now()
PROJECT_ID = "bt-int-ml-specialization"
PROJECT_NR = "738673379845"
REGION = "europe-west3"
CURRENT_DIR = Path(__file__).parent

DEPLOY_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest"
ENDPOINT_ID = "2009753323946639360"
ENDPOINT_NAME = f"projects/{PROJECT_NR}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
ENDPOINT_URI = f"https://{REGION}-aiplatform.googleapis.com/v1/{ENDPOINT_NAME}"
JOB_ID = f"demo1-{now:%Y-%m-%d-%H-%M-%S}"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-ml-demo1"
PUBSUB_SINK_TOPIC = f"projects/{PROJECT_ID}/topics/demo1-event-sink"
PUBSUB_SOURCE_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/demo1-event-source-subscription"
SUBNET = f"https://www.googleapis.com/compute/v1/projects/{PROJECT_ID}/regions/{REGION}/subnetworks/default-{REGION}"
TRANSFORM_ARTIFACT_LOCATION = f"gs://{PROJECT_ID}_dataflow_demo1/transform_artifacts/demo1-2024-08-26-09-38-51"


@dsl.pipeline(
    name="Vertex AI demo1",
    description="Vertex AI demo1",
)
def pipeline(
    display_name: str = "demo1",
    max_trial_count: int = 5,
    parallel_trial_count: int = 5,
    base_output_directory: str = PIPELINE_ROOT,
):
    """Pipeline to train a custom model on the chicago taxi driver dataset."""

    dataflow_train_batch_op = DataflowFlexTemplateJobOp(
        project=PROJECT_ID,
        location=REGION,
        container_spec_gcs_path=f"gs://{PROJECT_ID}_dataflow_demo1/templates/demo1-batch.json",
        job_name="batchpreprocess-train",
        num_workers=1,
        max_workers=1,
        service_account_email=f"d1-dataflow-batch-runner@{PROJECT_ID}.iam.gserviceaccount.com",
        temp_location=f"gs://{PROJECT_ID}_dataflow_demo1/batch/temp",
        machine_type="n1-standard-2",
        subnetwork=SUBNET,
        staging_location=f"gs://{PROJECT_ID}_dataflow_demo1/batch/staging",
        parameters={
            "project_id": PROJECT_ID,
            "df_run": JOB_ID,
            "output_location": f"gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/{JOB_ID}",
        },
        ip_configuration="WORKER_IP_PRIVATE",
    )

    dataflow_train_wait_op = WaitGcpResourcesOp(gcp_resources=dataflow_train_batch_op.outputs["gcp_resources"])

    dataflow_eval_batch_op = DataflowFlexTemplateJobOp(
        project=PROJECT_ID,
        location=REGION,
        container_spec_gcs_path=f"gs://{PROJECT_ID}_dataflow_demo1/templates/demo1-eval.json",
        job_name="batchpreprocess-eval",
        num_workers=1,
        max_workers=1,
        service_account_email=f"d1-dataflow-batch-runner@{PROJECT_ID}.iam.gserviceaccount.com",
        temp_location=f"gs://{PROJECT_ID}_dataflow_demo1/eval/temp",
        machine_type="n1-standard-2",
        subnetwork=SUBNET,
        staging_location=f"gs://{PROJECT_ID}_dataflow_demo1/eval/staging",
        parameters={
            "project_id": PROJECT_ID,
            "df_run": JOB_ID,
            "transform_artifact_location": TRANSFORM_ARTIFACT_LOCATION,
        },
        ip_configuration="WORKER_IP_PRIVATE",
    ).after(dataflow_train_wait_op)

    dataflow_eval_wait_op = WaitGcpResourcesOp(gcp_resources=dataflow_eval_batch_op.outputs["gcp_resources"])

    tuning_op, study_spec_metrics = components.get_hyperparametertuning_op(
        project_id=PROJECT_ID,
        job_id=JOB_ID,
        region=REGION,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        base_output_directory=base_output_directory,
    )

    tuning_op.after(dataflow_train_wait_op)

    best_trial_op = components.get_best_trial_op(
        gcp_resources=tuning_op.outputs["gcp_resources"], study_spec_metrics=study_spec_metrics
    )

    model_dir_op = components.get_model_dir(
        base_output_directory=base_output_directory, best_trial=best_trial_op.outputs["Output"]
    )

    unmanaged_model_importer = dsl.importer(
        artifact_uri=model_dir_op.outputs["Output"],
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={"containerSpec": {"imageUri": DEPLOY_IMAGE}},
    )

    model_upload_op = ModelUploadOp(
        display_name=display_name,
        unmanaged_container_model=unmanaged_model_importer.outputs["artifact"],
        project=PROJECT_ID,
        location=REGION,
    )

    batch_prediction_op = ModelBatchPredictOp(
        job_display_name="batch_prediction_for_eval",
        model=model_upload_op.outputs["model"],
        location=REGION,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source_uris=[f"gs://bt-int-ml-specialization_dataflow_demo1/jsonl_files/{JOB_ID}.jsonl"],
        instance_type="tf-record",
        included_fields=["dense_input"],
        gcs_destination_output_uri_prefix="gs://bt-int-ml-specialization-ml-demo1/",
        machine_type="n1-standard-2",
        max_replica_count=1,
        generate_explanation=False,
        project=PROJECT_ID,
    ).after(dataflow_eval_wait_op)

    _ = ModelEvaluationRegressionOp(
        target_field_name="fare",
        model=model_upload_op.outputs["model"],
        location=REGION,
        predictions_format="jsonl",
        predictions_gcs_source=batch_prediction_op.outputs["gcs_output_directory"],
        ground_truth_format="jsonl",
        prediction_score_column="prediction",
        dataflow_service_account="d1-dataflow-batch-runner@bt-int-ml-specialization.iam.gserviceaccount.com",
        dataflow_machine_type="n1-standard-2",
        dataflow_workers_num=1,
        dataflow_max_workers_num=1,
        dataflow_subnetwork=SUBNET,
        dataflow_use_public_ips=False,
        project=PROJECT_ID,
    )

    endpoint_importer = dsl.importer(
        artifact_uri=ENDPOINT_URI,
        artifact_class=artifact_types.VertexEndpoint,
        metadata={"resourceName": ENDPOINT_NAME},
    )

    model_deploy_op = ModelDeployOp(
        model=model_upload_op.outputs["model"],
        endpoint=endpoint_importer.output,
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        service_account=f"ml-demo1-predictor@{PROJECT_ID}.iam.gserviceaccount.com",
    )

    _ = DataflowFlexTemplateJobOp(
        enable_streaming_engine=True,
        ip_configuration="WORKER_IP_PRIVATE",
        num_workers=1,
        max_workers=1,
        parameters={
            "project_id": PROJECT_ID,
            "pubsub_sink_topic": PUBSUB_SINK_TOPIC,
            "pubsub_source_subscription": PUBSUB_SOURCE_SUBSCRIPTION,
            "transform_artifact_location": TRANSFORM_ARTIFACT_LOCATION,
            "endpoint_name": ENDPOINT_NAME,
        },
        project=PROJECT_ID,
        location=REGION,
        service_account_email=f"d1-dataflow-inference-runner@{PROJECT_ID}.iam.gserviceaccount.com",
        container_spec_gcs_path=f"gs://{PROJECT_ID}_dataflow_demo1/templates/demo1-inference.json",
        job_name="demo1-inference",
        staging_location=f"gs://{PROJECT_ID}_dataflow_demo1/inference/staging",
        subnetwork=SUBNET,
        temp_location=f"gs://{PROJECT_ID}_dataflow_demo1/inference/temp",
        machine_type="n1-standard-2",
    ).after(dataflow_train_wait_op)


def run_pipeline():
    pipeline_file_path = str(CURRENT_DIR / "pipeline.json")
    compiler.Compiler().compile(pipeline, pipeline_file_path)  # type: ignore

    run = aiplatform.PipelineJob(
        project=PROJECT_ID,
        location=REGION,
        display_name="demo1",
        template_path=pipeline_file_path,
        job_id=JOB_ID,
        enable_caching=False,
        pipeline_root=PIPELINE_ROOT,
    )

    aiplatform.init(project=PROJECT_ID, location=REGION, experiment="demo1-wine-exp")

    run.submit(
        service_account=f"ml-demo1-executor@{PROJECT_ID}.iam.gserviceaccount.com",
        experiment="demo1-exp",
    )
    return run


if __name__ == "__main__":
    run_pipeline()
