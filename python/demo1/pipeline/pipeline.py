"""
Vertex AI pipeline definition with kubeflow pipelines DSL.

This pipeline
  - loads an example dataset (wine from sklearn)
  - splits the dataset into train, validate and test
  - trains a custom model on the train dataset
"""


from datetime import datetime
from pathlib import Path
from google.cloud import aiplatform
from kfp import dsl, compiler
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import HyperparameterTuningJobRunOp
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from google_cloud_pipeline_components.v1.dataflow import DataflowFlexTemplateJobOp
from google_cloud_pipeline_components.v1.wait_gcp_resources import WaitGcpResourcesOp

now = datetime.now()
JOB_ID = f"demo1-{now:%Y-%m-%d-%H-%M-%S}"
PROJECT = "bt-int-ml-specialization"
REGION = "europe-west3"
PIPELINE_ROOT = f"gs://{PROJECT}-ml-demo1"
CURRENT_DIR = Path(__file__).parent
DEPLOY_IMAGE = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest"
PUBSUB_SINK_TOPIC = f"projects/{PROJECT}/topics/demo1-event-sink"
PUBSUB_SOURCE_SUBSCRIPTION = f"projects/{PROJECT}/subscriptions/demo1-event-source-subscription"
TRANSFORM_ARTIFACT_LOCATION = f"gs://{PROJECT}_dataflow_demo1/transform_artifacts/{JOB_ID}"
ENDPOINT_ID = '2134275214815526912'
PROJECT_ID = '738673379845'
ENDPOINT_NAME = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
ENDPOINT_URI = f"https://{REGION}-aiplatform.googleapis.com/v1/{ENDPOINT_NAME}"


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"])
def model_dir(base_output_directory: str, best_trial: str) -> str:
    from google.cloud.aiplatform_v1.types import study

    trial_proto = study.Trial.from_json(best_trial)
    model_id = trial_proto.id
    return f"{base_output_directory}/{model_id}/model"


@dsl.component(
    packages_to_install=['google-cloud-aiplatform',
                         'google-cloud-pipeline-components',
                         'protobuf'], base_image='python:3.10')
def GetBestTrialOp(gcp_resources: str, study_spec_metrics: list) -> str:
    from google.cloud import aiplatform
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
    from google.protobuf.json_format import Parse
    from google.cloud.aiplatform_v1.types import study

    api_endpoint_suffix = '-aiplatform.googleapis.com'
    gcp_resources_proto = Parse(gcp_resources, GcpResources())
    gcp_resources_split = gcp_resources_proto.resources[0].resource_uri.partition(
        'projects')
    resource_name = gcp_resources_split[1] + gcp_resources_split[2]
    prefix_str = gcp_resources_split[0]
    prefix_str = prefix_str[:prefix_str.find(api_endpoint_suffix)]
    api_endpoint = prefix_str[(prefix_str.rfind('//') + 2):] + api_endpoint_suffix

    client_options = {'api_endpoint': api_endpoint}
    job_client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    response = job_client.get_hyperparameter_tuning_job(name=resource_name)

    trials = [study.Trial.to_json(trial) for trial in response.trials]

    if len(study_spec_metrics) > 1:
        raise RuntimeError('Unable to determine best parameters for multi-objective'
                           ' hyperparameter tuning.')
    trials_list = [study.Trial.from_json(trial) for trial in trials]
    best_trial = None
    goal = study_spec_metrics[0]['goal']
    best_fn = None
    if goal == study.StudySpec.MetricSpec.GoalType.MAXIMIZE:
        best_fn = max
    elif goal == study.StudySpec.MetricSpec.GoalType.MINIMIZE:
        best_fn = min
    best_trial = best_fn(
        trials_list, key=lambda trial: trial.final_measurement.metrics[0].value)

    return study.Trial.to_json(best_trial)


@dsl.pipeline(
    name='Vertex AI demo1',
    description='Vertex AI demo1',
)
def pipeline(display_name: str = "demo1",
             max_trial_count: int = 1,
             parallel_trial_count: int = 1,
             base_output_directory: str = PIPELINE_ROOT,
             project: str = PROJECT,
             region: str = REGION,
             ):
    """Pipeline to train a custom model on the chicago taxi driver dataset."""

    # Launch the Dataflow Flex Template job
    dataflow_batch_op = DataflowFlexTemplateJobOp(
        project=PROJECT,
        location=REGION,
        container_spec_gcs_path=f'gs://{PROJECT}_dataflow_demo1/templates/demo1-batch.json',
        job_name='batchpreprocess',
        num_workers=1,
        max_workers=1,
        service_account_email=f"d1-dataflow-batch-runner@{PROJECT}.iam.gserviceaccount.com",
        temp_location=f"gs://{PROJECT}_dataflow_demo1/batch/temp",
        machine_type="n1-standard-2",
        subnetwork=f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{REGION}/subnetworks/default-{REGION}",
        staging_location=f"gs://{PROJECT}_dataflow_demo1/batch/staging",
        parameters={'project_id': PROJECT,
                    'df_run': JOB_ID},
        ip_configuration='WORKER_IP_PRIVATE'
    )

    dataflow_wait_op = WaitGcpResourcesOp(
        gcp_resources=dataflow_batch_op.outputs["gcp_resources"]
    )

    command = [
        "python",
        "/app/main.py"
    ]
    args = [
        "--train_file_path",
        #"gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-02-08T17:04:29.494733-00000-of-00001.tfrecord"
        "gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/" + JOB_ID + "-00000-of-00001.tfrecord"
    ]

    # The spec of the worker pools including machine type and Docker image
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                # "accelerator_type": "NVIDIA_TESLA_T4",
                # "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/train:latest",
                "command": command,
                "args": args},
        }
    ]

    # List serialized from the dictionary representing metrics to optimize.
    # The dictionary key is the metric_id, which is reported by your training job,
    # and the dictionary value is the optimization goal of the metric.
    study_spec_metrics = hyperparameter_tuning_job.serialize_metrics({"loss": "minimize"})

    # List serialized from the parameter dictionary. The dictionary
    # represents parameters to optimize. The dictionary key is the parameter_id,
    # which is passed into your training job as a command line key word argument, and the
    # dictionary value is the parameter specification of the metric.
    study_spec_parameters = hyperparameter_tuning_job.serialize_parameters(
        {
            "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                min=0.001, max=1, scale="log"
            ),
            "dropout_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                min=0.05, max=0.3, scale="linear"
            ),
        }
    )

    tuning_op = HyperparameterTuningJobRunOp(
        display_name="hpt_demo1",
        project=project,
        location=region,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=study_spec_metrics,
        study_spec_parameters=study_spec_parameters,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        base_output_directory=base_output_directory,
    )

    tuning_op.after(dataflow_wait_op)

    best_trial_op = GetBestTrialOp(
        gcp_resources=tuning_op.outputs["gcp_resources"], study_spec_metrics=study_spec_metrics
    )

    model_dir_op = model_dir(base_output_directory=base_output_directory,
                             best_trial=best_trial_op.outputs['Output'])

    unmanaged_model_importer = dsl.importer(
        artifact_uri=model_dir_op.outputs['Output'],
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            'containerSpec': {
                'imageUri': DEPLOY_IMAGE
            }
        }
    )

    model_upload_op = ModelUploadOp(
        display_name=display_name,
        unmanaged_container_model=unmanaged_model_importer.outputs['artifact'],
        project=PROJECT,
        location=REGION,
    )

    # todo: {model_version} add model version to display name

    endpoint_importer = dsl.importer(
        artifact_uri=ENDPOINT_URI,
        artifact_class=artifact_types.VertexEndpoint,
        metadata={
            "resourceName": ENDPOINT_NAME
        }
    ).output

    model_deploy_op = ModelDeployOp(
        model=model_upload_op.outputs["model"],
        endpoint=endpoint_importer,
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
        service_account=f"ml-demo1-predictor@{PROJECT}.iam.gserviceaccount.com",
    )

    # Launch the Dataflow Flex Template job
    dataflow_inf_op = DataflowFlexTemplateJobOp(
        enable_streaming_engine=True,
        ip_configuration='WORKER_IP_PRIVATE',
        num_workers=1,
        max_workers=1,
        parameters={'project_id': PROJECT,
                    'pubsub_sink_topic': PUBSUB_SINK_TOPIC,
                    'pubsub_source_subscription': PUBSUB_SOURCE_SUBSCRIPTION,
                    'transform_artifact_location': TRANSFORM_ARTIFACT_LOCATION,
                    'endpoint_name': ENDPOINT_NAME},
        project=PROJECT,
        location=REGION,
        service_account_email=f"d1-dataflow-inference-runner@{PROJECT}.iam.gserviceaccount.com",
        container_spec_gcs_path=f'gs://{PROJECT}_dataflow_demo1/templates/demo1-inference.json',
        job_name='demo1-inference',
        staging_location=f"gs://{PROJECT}_dataflow_demo1/inference/staging",
        subnetwork=f"https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{REGION}/subnetworks/default-{REGION}",
        temp_location=f"gs://{PROJECT}_dataflow_demo1/inference/temp",
        machine_type="n1-standard-2"
    )

    dataflow_inf_op.after(model_deploy_op)

    # todo: use parent model to create different versions of model
    # todo: prio3: white paper
    # todo: prio4:metrics beim Training
    # todo: readme file for repo


def run_pipeline():
    pipeline_file_path = str(CURRENT_DIR / "pipeline.json")
    compiler.Compiler().compile(pipeline, pipeline_file_path)

    run = aiplatform.PipelineJob(
        project=PROJECT,
        location=REGION,
        display_name="demo1",
        template_path=pipeline_file_path,
        job_id=JOB_ID,
        enable_caching=True,
        pipeline_root=PIPELINE_ROOT,
    )

    aiplatform.init(project=PROJECT, location=REGION, experiment="demo1-wine-exp")

    run.submit(
        service_account=f"ml-demo1-executor@{PROJECT}.iam.gserviceaccount.com",
        experiment="demo1-exp",
    )
    return run


if __name__ == "__main__":
    run_pipeline()
