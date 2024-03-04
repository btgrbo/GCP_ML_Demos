from kfp import dsl

components = ["pyarrow==15.0.0", "pandas==2.2.1", "scikit-learn==1.4.1.post1", "matplotlib==3.8.3"]


@dsl.component(base_image="python:3.10", packages_to_install=components)
def get_metrics(
    predictions: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    prediction_error_curve: dsl.OutputPath("png"),
):

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.metrics import PredictionErrorDisplay, mean_squared_error, r2_score

    df = pd.read_parquet(predictions.path)

    fig, ax = plt.subplots()
    PredictionErrorDisplay.from_predictions(
        y_true=df.Purchase,
        y_pred=df.Prediction,
        ax=ax,
        kind="actual_vs_predicted",
    )
    fig.savefig(prediction_error_curve)

    metrics.log_metric("RMSE:", float(np.sqrt(mean_squared_error(df.Purchase, y_pred=df.Prediction))))
    metrics.log_metric("y_mean", float(df.Purchase.mean()))
    metrics.log_metric("y_predicted_mean", float(df.Prediction.mean()))
    metrics.log_metric("R2 Score:", float(np.round(r2_score(df.Purchase, y_pred=df.Prediction) * 100, 2)))
