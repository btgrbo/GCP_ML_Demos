import xgboost as xgb
import pandas as pd

from pathlib import Path
from fire import Fire

# Main function for training execution
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

    # Load and transform training data
    train = pd.read_parquet(train_file_parquet)

    # Split the data into features and target
    X_train = train.drop(columns=['purchase'])
    y_train = train['purchase']

    # Create and train XGB model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    model.save_model(model_file)
    print(f"Model saved to {model_file}")

    # Load and transform evaluation data
    eval_data = pd.read_parquet(eval_file_parquet)

    # Make predictions on the evaluation data
    eval_predictions = model.predict(eval_data.drop(columns=['purchase']))
    eval_data['prediction'] = eval_predictions
    eval_data.to_parquet(eval_output_file_parquet, index=False)
    print(f"Eval saved to {eval_output_file_parquet}")

if __name__ == '__main__':
    # run with `python main.py \
    #   --train_file_parquet path/to/train.parquet \
    #   --eval_file_parquet path/to/eval.parquet \
    #   --model_file path/to/model.xgb \
    #   --eval_output_file_parquet path/to/eval_with_predictions.parquet`

    Fire(main)


#   docker build -t europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/train:latest .
#   docker run -it -v "C:\Users\OliverNowakbtelligen\OneDrive - b.telligent group\Desktop\GCP ML Demo\big-query_output.parquet":/m/data.parquet europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/train:latest python main.py /m/data.parquet /m/data.parquet ./test.pckl ./xyz.parquet

# cd .\python\demo2\   
# python main.py ./test/data.parquet ./test/data.parquet ./test/model.pckl ./test/xyz.parquet
# python main.py ./test/training.parquet ./test/training.parquet ./test/model_2.pckl ./test/wxyz.parquet

# docker images 
# pip freeze

# Git commands
# cd "C:\Users\OliverNowakbtelligen\OneDrive - b.telligent group\Desktop\Tickets"
# git clone git@github.com:btgrbo/GCP_ML_Demos.git