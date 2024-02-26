import pickle
from pathlib import Path

import pandas as pd
from fire import Fire
from sklearn.ensemble import RandomForestRegressor


def save_model(model, file: str):
    with open(file, "wb") as f:
        pickle.dump(model, f)


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
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    save_model(model, model_file)
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
