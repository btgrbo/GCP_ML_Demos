from kfp import dsl

components = ["scikit-learn==1.4.1.post1", "pyarrow==15.0.0", "pandas==2.2.1"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def split_data(
    data: dsl.Input[dsl.Dataset],
    split_ratio: float,
    split_a: dsl.Output[dsl.Dataset],
    split_b: dsl.Output[dsl.Dataset],
):

    import pandas as pd
    from sklearn.model_selection import train_test_split as tts

    all_data = pd.read_parquet(data.path)

    a, b = tts(all_data, test_size=split_ratio)

    a.to_parquet(split_a.path, index=False)
    b.to_parquet(split_b.path, index=False)
