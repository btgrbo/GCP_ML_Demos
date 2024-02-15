import xgboost as xgb
import pandas as pd

from pathlib import Path
from fire import Fire
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class BlackFridayTransformer(TransformerMixin):
    """Class for dynamic input of the black friday (sales data)"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Filling NULL values in Product_Category_2 and Product_Category_3
        X['Product_Category_2_new'] = X['Product_Category_2'].fillna(1)
        X['Product_Category_3_new'] = X['Product_Category_3'].fillna(2)
        
        # Mapping gender to a numerical value
        X['Gender_New'] = X['Gender'].map({'M': 0, 'F': 1})
        
        # One-Hot Encoding for Age groups
        # Fit für die entsprechende Variable anwenden, bevor transform ausgeführt wird
        # In das bestehende Framework mit einbauen

        age_encoder = OneHotEncoder(drop='first', sparse_output=False)
        age_encoded = age_encoder.fit_transform(X[['Age']])
        age_columns = [f'age_{age}' for age in age_encoder.categories_[0][1:]]
        X[age_columns] = pd.DataFrame(age_encoded, columns=age_columns, index=X.index)
        
        # One-Hot Encoding for City categories
        city_encoder = OneHotEncoder(drop='first', sparse_output=False)
        city_encoded = city_encoder.fit_transform(X[['City_Category']])
        city_columns = [f'city_category_{city}' for city in city_encoder.categories_[0][1:]]
        X[city_columns] = pd.DataFrame(city_encoded, columns=city_columns, index=X.index)
        
        # One-Hot Encoding for Stay in city
        stay_encoder = OneHotEncoder(drop='first', sparse_output=False)
        stay_encoded = stay_encoder.fit_transform(X[['Stay_In_Current_City_Years']])
        stay_columns = [f'stay_{years}_years' for years in stay_encoder.categories_[0][1:]]
        X[stay_columns] = pd.DataFrame(stay_encoded, columns=stay_columns, index=X.index)
        
        # Renaming columns
        X = X.rename(columns={
            'Purchase': 'purchase',
            'Occupation': 'occupation',
            'Marital_Status': 'marital_status',
            'Gender_New': 'gender',
            'Product_Category_1': 'product_category_1',
            'Product_Category_2_new': 'product_category_2',
            'Product_Category_3_new': 'product_category_3',
        })

        # Selecting relevant columns
        selected_columns = [
            'purchase', 'occupation', 'marital_status', 'gender',
            'product_category_1', 'product_category_2', 'product_category_3',
            *age_columns, *city_columns, *stay_columns
        ]
        
        return X[selected_columns]


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
    transformer = BlackFridayTransformer()
    train_transformed = transformer.transform(train)

    # Split the transformed data into features and target
    X_train = train_transformed.drop(columns=['purchase'])
    y_train = train_transformed['purchase']

    # Create and train XGB model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    model.save_model(model_file)
    print(f"Model saved to {model_file}")

    # Load and transform evaluation data
    eval_data = pd.read_parquet(eval_file_parquet)
    eval_transformed = transformer.transform(eval_data)

    # Make predictions on the transformed evaluation data
    eval_predictions = model.predict(eval_transformed.drop(columns=['purchase']))
    eval_transformed['prediction'] = eval_predictions
    eval_transformed.to_parquet(eval_output_file_parquet, index=False)
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