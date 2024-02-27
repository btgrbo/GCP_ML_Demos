from google_cloud_pipeline_components.types import artifact_types
from kfp import dsl

components = ["google-cloud-pipeline-components"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def import_model(
    model_artifact: dsl.Input[dsl.Model],
    artifact: dsl.Output[artifact_types.UnmanagedContainerModel],
):

    artifact.uri = ""
    artifact.metadata = {
        "containerSpec": {
            "imageUri": "europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/prediction:latest",
            "healthRoute": "/ping",
            "predictRoute": "/predictions",
            "env": [{"name": "MODEL_URI", "value": model_artifact.uri}],
        }
    }

    # using artifact.uri that can be uset in container as AIP_STORAGE_URI is making trouble due to missing permissions
