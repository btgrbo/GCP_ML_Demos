from pathlib import Path

import xgboost as xgb
import pandas as pd
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

    # Remove the ID columns before creating a ML model
    train = train.drop(['User_ID', 'Product_ID'], axis=1)

    # Splitting the daza into features and target
    X_train = train.drop('Purchase', axis=1)
    y_train = train['Purchase']

    # Create simple XGB Model
    model = xgb.XGBClassifier()

    # Fitting defined model
    model.fit(X_train, y_train)
    model.save_model(model_file)
    print(f"Model saved to {model_file}")

    # Make predictions on test dataset
    eval = pd.read_parquet(eval_file_parquet)
    eval_data = eval_data.drop(['User_ID', 'Product_ID'], axis=1)
    yhat = model.predict(eval_data)

    # Save predictions
    eval['prediction'] = yhat
    eval.to_parquet(eval_output_file_parquet)
    print(f"Eval saved to {eval_output_file_parquet}")

if __name__ == '__main__':
    # run with `python demo2/main.py \
    #   --train_file_parquet demo2/train.parquet \
    #   --eval_file_parquet demo2/eval.parquet \
    #   --model_file demo2/model.xgb \
    #   --eval_output_file_parquet demo2/eval_with_predictions.parquet`

    Fire(main)