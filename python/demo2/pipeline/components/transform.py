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
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.base import TransformerMixin
    import pandas as pd
    import pickle
    from pathlib import Path

    class BlackFridayTransformer(TransformerMixin):
        """Class for dynamic input of the black Friday (sales data)"""

        def __init__(self):
            self.age_columns = None
            self.city_columns = None
            self.stay_columns = None
            self.age_encoder = OneHotEncoder(drop='first', sparse_output=False)
            self.city_encoder = OneHotEncoder(drop='first', sparse_output=False)
            self.stay_encoder = OneHotEncoder(drop='first', sparse_output=False)

        def fit(self, X, y=None):
            # Filling NULL values in Product_Category_2 and Product_Category_3
            X['Product_Category_2_new'] = X['Product_Category_2'].fillna(1)
            X['Product_Category_3_new'] = X['Product_Category_3'].fillna(2)

            # Mapping gender to a numerical value
            X['Gender_New'] = X['Gender'].map({'M': 0, 'F': 1})

            # Fit and transform Age
            age_encoded = self.age_encoder.fit_transform(X[['Age']])
            self.age_columns = [f'age_{age}' for age in self.age_encoder.categories_[0][1:]]
            X[self.age_columns] = pd.DataFrame(age_encoded, columns=self.age_columns, index=X.index)

            # Fit and transform City categories
            city_encoded = self.city_encoder.fit_transform(X[['City_Category']])
            self.city_columns = [f'city_category_{city}' for city in self.city_encoder.categories_[0][1:]]
            X[self.city_columns] = pd.DataFrame(city_encoded, columns=self.city_columns, index=X.index)

            # Fit and transform Stay in city
            stay_encoded = self.stay_encoder.fit_transform(X[['Stay_In_Current_City_Years']])
            self.stay_columns = [f'stay_{years}_years' for years in self.stay_encoder.categories_[0][1:]]
            X[self.stay_columns] = pd.DataFrame(stay_encoded, columns=self.stay_columns, index=X.index)

            return self

        def transform(self, X):
            # Apply transformations based on the fitted encoders
            X['Product_Category_2_new'] = X['Product_Category_2'].fillna(1)
            X['Product_Category_3_new'] = X['Product_Category_3'].fillna(2)
            X['Gender_New'] = X['Gender'].map({'M': 0, 'F': 1})

            age_encoded = self.age_encoder.transform(X[['Age']])
            X[self.age_columns] = pd.DataFrame(age_encoded, columns=self.age_columns, index=X.index)

            city_encoded = self.city_encoder.transform(X[['City_Category']])
            X[self.city_columns] = pd.DataFrame(city_encoded, columns=self.city_columns, index=X.index)

            stay_encoded = self.stay_encoder.transform(X[['Stay_In_Current_City_Years']])
            X[self.stay_columns] = pd.DataFrame(stay_encoded, columns=self.stay_columns, index=X.index)

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
                *self.age_columns, *self.city_columns, *self.stay_columns
            ]

            return X[selected_columns]

    data = pd.read_parquet(data.path)

    # black_friday_transformer = BlackFridayTransformer()
    # data = black_friday_transformer.transform(data)
    
    pipe = ColumnTransformer(
        [
            # ('scaler', StandardScaler(), ['x']),
            ('black_friday_transformer', BlackFridayTransformer(), data.columns),
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
