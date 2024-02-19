from pathlib import Path

import pandas as pd
import xgboost as xgb
from fire import Fire


def main(
    train_file_parquet: str,
    eval_file_parquet: str,
    model_file: str,
    eval_output_file_parquet: str,
):

    model_file = Path(model_file)
    model_file.parent.mkdir(exist_ok=True, parents=True)

    eval_output_file_parquet = Path(eval_output_file_parquet)
    eval_output_file_parquet.parent.mkdir(exist_ok=True, parents=True)

    train = pd.read_parquet(train_file_parquet)

    # Split the transformed data into features and target
    X_train = train.drop(columns=["Purchase"])
    y_train = train["Purchase"]

    # Create and train XGB model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    model.save_model(model_file)
    print(f"Model saved to {model_file}")

    # Load and transform evaluation data
    eval = pd.read_parquet(eval_file_parquet)

    # Make predictions on the transformed evaluation data
    eval_predictions = model.predict(eval.drop(columns=["Purchase"]))
    eval["Prediction"] = eval_predictions
    eval.to_parquet(eval_output_file_parquet, index=False)
    print(f"Eval saved to {eval_output_file_parquet}")


if __name__ == "__main__":
    Fire(main)
