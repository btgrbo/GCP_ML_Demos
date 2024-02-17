from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from kfp import dsl

components = ["google-cloud-pipeline-components"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def import_model(
        model_artifact: dsl.Input[dsl.Model],
        model: dsl.Output[artifact_types.UnmanagedContainerModel],
):
    model.artifact_uri = model_artifact.uri
    model.metadata = {
        'containerSpec': {
            'imageUri': "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
        }
    }

    CustomTrainingJobOp()
