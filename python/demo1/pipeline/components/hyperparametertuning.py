from google.cloud import aiplatform
from google_cloud_pipeline_components.v1 import hyperparameter_tuning_job


def get_hyperparametertuning_op(
    project_id: str,
    job_id: str,
    region: str,
    max_trial_count,
    parallel_trial_count,
    base_output_directory,
):

    # The spec of the worker pools including machine type and Docker image
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-16",
                #  "accelerator_type": "NVIDIA_TESLA_T4",
                #  "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/train:tensorboard",
                "command": ["python", "/app/main.py"],
                # "args": ["--train_file_path", f"gs://{project_id}_dataflow_demo1/TFRecords/{job_id}/"],
                "args": [
                    "--train_file_path",
                    "gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/demo1-2024-03-14-14-11-42/",
                ],
            },
        }
    ]

    study_spec_metrics = hyperparameter_tuning_job.serialize_metrics({"loss": "minimize"})

    study_spec_parameters = hyperparameter_tuning_job.serialize_parameters(
        {
            "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0.001, max=1, scale="log"),
            "dropout_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(min=0.05, max=0.3, scale="linear"),
        }
    )

    return (
        hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
            display_name="hpt_demo1",
            project=project_id,
            location=region,
            worker_pool_specs=worker_pool_specs,
            study_spec_metrics=study_spec_metrics,
            study_spec_parameters=study_spec_parameters,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            base_output_directory=base_output_directory,
        ),
        study_spec_metrics,
    )
