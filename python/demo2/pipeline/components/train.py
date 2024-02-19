from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.container_component
def training(
    train_file_parquet: Input[Dataset],
    eval_file_parquet: Input[Dataset],
    model: Output[Model],
    eval_output_file_parquet: Output[Dataset],
):
    return dsl.ContainerSpec(
        image="europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/train:latest",
        command=[
            "python",
            "/app/main.py",
        ],
        args=[train_file_parquet.path, eval_file_parquet.path, model.path, eval_output_file_parquet.path],
    )
