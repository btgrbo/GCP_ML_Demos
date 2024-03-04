from kfp import dsl

components = ["scikit-learn==1.4.1.post1", "pyarrow==15.0.0", "pandas==2.2.1"]


@dsl.component(
    base_image="python:3.10", packages_to_install=components)
def transform(
        data: dsl.Input[dsl.Dataset],
        data_proc: dsl.Output[dsl.Dataset],
        preprocessor: dsl.Output[dsl.Model],
):
    preprocessor.framework = "sklearn"

    import pickle
    from pathlib import Path

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    df = pd.read_parquet(data.path)

    y = df["Purchase"]
    df = df.drop(columns=["Purchase"])

    drop_features = ["Product_Category_3", "Product_ID", "User_ID"]
    categorical_features = ["Age", "City_Category", "Gender", "Product_Category_1", "Product_Category_2"]
    ordinal_features = ["Age", "Stay_In_Current_City_Years"]

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one-hot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    pipe = ColumnTransformer(
        [
            ("drop_columns", "drop", drop_features),
            ("categorical pipeline", categorical_pipeline, categorical_features),
            ("ordinal_encoding", OrdinalEncoder(), ordinal_features),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    out = pipe.fit_transform(df)
    out = pd.DataFrame(data=out, columns=pipe.get_feature_names_out())

    out["Purchase"] = y

    Path(preprocessor.path).parent.mkdir(parents=True, exist_ok=True)
    with open(preprocessor.path, "wb") as f:
        pickle.dump(pipe, f)

    Path(data_proc.path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(data_proc.path, index=False)


# if __name__ == "__main__":
#     from io import BytesIO
#     from unittest.mock import Mock

#     import pandas as pd

#     df = pd.DataFrame(
#         {
#             "foo": range(3),
#             "User_ID": range(3),
#             "Product_ID": range(3),
#             "Product_Category_1": range(3),
#             "Product_Category_2": range(3),
#             "Product_Category_3": range(3),
#             "Gender": [str(i) for i in range(3)],
#             "Age": [str(i) for i in range(3)],
#             "City_Category": [str(i) for i in range(3)],
#             "Stay_In_Current_City_Years": [str(i) for i in range(3)],
#             "Purchase": range(3),
#         }
#     )

#     buf = BytesIO()
#     df.to_parquet(buf)
#     buf.seek(0)

#     data = Mock()
#     data.path = buf

#     transform.python_func(Mock(path=buf), Mock(path="/tmp/foo1"), Mock(path="/tmp/foo1"))
