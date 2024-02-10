from kfp import dsl

components = ["scikit-learn", "pyarrow", "pandas"]


@dsl.component(
    base_image="python:3.10", packages_to_install=components)
def transform(
        data: dsl.Input[dsl.Dataset],
        data_proc: dsl.Output[dsl.Dataset],
        preprocessor: dsl.Output[dsl.Model],
):
    preprocessor.framework = "sklearn"

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import pickle
    from pathlib import Path

    data = pd.read_parquet(data.path)

    pipe = ColumnTransformer(
        [
            # ('scaler', StandardScaler(), ['x']),
            ('drop_columns', 'drop', ['User_ID', 'Product_ID'])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    out = pipe.fit_transform(data)
    out = pd.DataFrame(data=out, columns=pipe.get_feature_names_out())

    Path(preprocessor.path).parent.mkdir(parents=True, exist_ok=True)
    with open(preprocessor.path, "wb") as f:
        pickle.dump(pipe, f)

    Path(data_proc.path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(data_proc.path)


# if __name__ == "__main__":
#     import pandas as pd
#     from io import BytesIO
#     from unittest.mock import Mock
#
#     df = pd.DataFrame({"foo": range(3), "User_ID": range(3), "Product_ID": range(3)})
#
#     buf = BytesIO()
#     df.to_parquet(buf)
#     buf.seek(0)
#
#     data = Mock()
#     data.path = buf
#
#     transform.python_func(Mock(path=buf), Mock(path="/tmp/foo1"),
#                           Mock(path="/tmp/foo1"))
