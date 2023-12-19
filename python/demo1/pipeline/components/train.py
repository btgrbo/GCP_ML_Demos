from kfp import dsl
from kfp.dsl import Input, Dataset


@dsl.container_component
def training(
    train_file_parquet: Input[Dataset],
    eval_file_parquet: Input[Dataset],
):
    return dsl.ContainerSpec(
        image="europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/train:latest",
        command=[
            "python",
            "/app/main.py",
        ],
        args=[
            train_file_parquet.path,
            eval_file_parquet.path,
        ],
    )
