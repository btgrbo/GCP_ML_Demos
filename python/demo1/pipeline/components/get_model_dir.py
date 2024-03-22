from kfp import dsl


@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"])
def get_model_dir(base_output_directory: str, best_trial: str) -> str:
    from google.cloud.aiplatform_v1.types import study

    trial_proto = study.Trial.from_json(best_trial)
    model_id = trial_proto.id
    return f"{base_output_directory}/{model_id}/model"
