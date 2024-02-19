from google_cloud_pipeline_components.types import artifact_types
from kfp import dsl

components = ["google-cloud-pipeline-components"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def import_model(
    model_artifact: dsl.Input[dsl.Model],
    artifact: dsl.Output[artifact_types.UnmanagedContainerModel],
):

    parent_uri = "/".join(model_artifact.uri.split("/")[0:-1])
    artifact.uri = parent_uri
    artifact.metadata = {
        "containerSpec": {
            "imageUri": "europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/xgboost-cpu.2-0:latest"
        }
    }
