from kfp import dsl

components = ["scikit-learn", "xgboost"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def create_pipeline(
    preprocessor: dsl.Input[dsl.Model],
    model: dsl.Input[dsl.Model],
    pipeline: dsl.Output[dsl.Model],
):
    pipeline.framework = "sklearn"
    pipeline.uri = pipeline.uri.replace("/pipeline", "/model.pkl")  # needed by xgboost container

    import pickle

    import xgboost
    from sklearn.pipeline import Pipeline

    with open(preprocessor.path, "rb") as f:
        preprocessor = pickle.load(f)

    model_ = xgboost.XGBRegressor()
    model_.load_model(model.path)

    pipe = Pipeline([("preprocess", preprocessor), ("model", model_)])

    with open(pipeline.path, "wb") as f:
        pickle.dump(pipe, f)
