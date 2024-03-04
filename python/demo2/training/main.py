import pickle
from pathlib import Path

import pandas as pd
from fire import Fire
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def save_model(model, file: str):
    with open(file, "wb") as f:
        pickle.dump(model, f)


def main(
    train_file_parquet: str,
    eval_file_parquet: str,
    model_file: str,
    eval_output_file_parquet: str,
    train_output_file_parquet: str,
    hyperparameter_tuning: bool = False,
):

    model_file = Path(model_file)
    model_file.parent.mkdir(exist_ok=True, parents=True)

    eval_output_file_parquet = Path(eval_output_file_parquet)
    eval_output_file_parquet.parent.mkdir(exist_ok=True, parents=True)

    train_output_file_parquet = Path(train_output_file_parquet)
    train_output_file_parquet.parent.mkdir(exist_ok=True, parents=True)

    train = pd.read_parquet(train_file_parquet)

    # Split the transformed data into features and target
    X_train = train.drop(columns=["Purchase"])
    y_train = train["Purchase"]

    # best hyperparameters from random search
    hyperparams = {
        "n_estimators": 200,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": "log2",
        "max_depth": None,
    }

    # Create and train XGB model
    print("Starting training...")
    model = RandomForestRegressor(**hyperparams)
    model.fit(X_train, y_train)
    save_model(model, model_file)
    print(f"Model saved to {model_file}")

    # --------------------------------------------------------------------
    # Random Search
    print("performing hyperparameter tuning")
    if hyperparameter_tuning:
        param_dist = {
            "n_estimators": [10, 50, 100, 200],  # Number of trees
            "max_depth": [None, 10, 20, 30],  # Maximum depth of each tree
            "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
            "min_samples_leaf": [1, 2, 4],  # Minimum samples required at a leaf node
            "max_features": ["sqrt", "log2", None],  # Maximum features to consider for splitting
        }

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,  # Number of random combinations to try
            cv=5,  # Cross-validation folds
            verbose=2,
            n_jobs=-1,  # Use all available CPU cores
        )
        random_search.fit(X_train, y_train)

        print("Best hyperparameters:", random_search.best_params_)
        model = random_search.best_estimator_
    # --------------------------------------------------------------------

    # Save training results
    train_predictions = pd.Series(model.predict(X_train), name="Prediction")
    pd.concat([train_predictions, y_train], axis=1).to_parquet(train_output_file_parquet, index=False)

    # Load and transform evaluation data
    eval = pd.read_parquet(eval_file_parquet)

    # Make predictions on the transformed evaluation data
    eval_predictions = model.predict(eval.drop(columns=["Purchase"]))
    eval["Prediction"] = eval_predictions
    eval.to_parquet(eval_output_file_parquet, index=False)
    print(f"Eval saved to {eval_output_file_parquet}")


if __name__ == "__main__":
    Fire(main)
