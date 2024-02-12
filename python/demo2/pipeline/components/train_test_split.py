from kfp import dsl


components = ["scikit-learn", "pyarrow", "pandas"]

@dsl.component(
    base_image="python:3.10", packages_to_install=components)
def train_test_split(
        data: dsl.Input[dsl.Dataset],
        test_ratio: float,
        data_train: dsl.Output[dsl.Dataset],
        data_test: dsl.Output[dsl.Dataset],
):


    from sklearn.model_selection import train_test_split as tts
    import pandas as pd

    all_data = pd.read_parquet(data.path)

    train, test = tts(all_data, test_size=test_ratio)

    train.to_parquet(data_train.path)
    test.to_parquet(data_test.path)



