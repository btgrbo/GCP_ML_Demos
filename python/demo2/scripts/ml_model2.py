import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading the data frame into the environment
df = pd.read_csv(r"C:/Users/OliverNowakbtelligen/OneDrive - b.telligent group/Desktop/GCP ML Demo/big-query_output.csv")

# Remove the ID columns before creating an ML model
df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Splitting the data into training and test sets
X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base regressors
base_regressors = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    # Add more base regressors if needed
]

# Define meta-regressor
meta_regressor = GradientBoostingRegressor(n_estimators=50, random_state=42)

# Create StackingRegressor
stacked_model = StackingRegressor(estimators=base_regressors, final_estimator=meta_regressor)

# Fit the model
stacked_model.fit(X_train, y_train)

# Make predictions on the test set
yhat_stacked = stacked_model.predict(X_test)

# Calculate MAE for the stacked model
mae_stacked = mean_absolute_error(y_test, yhat_stacked)
print('MAE for Stacked Model:', mae_stacked)

# Calculate R2 score for the stacked model
r2_stacked = r2_score(y_test, yhat_stacked)
print('R2 Score for Stacked Model:', r2_stacked)

# Calculate feature importance for the stacked model
if hasattr(stacked_model, 'final_estimator_') and hasattr(stacked_model.final_estimator_, 'feature_importances_'):
    feature_importance = stacked_model.final_estimator_.feature_importances_
    feature_names = X_train.columns

    # Sort feature importance and feature names
    sorted_indices = feature_importance.argsort()[::-1]
    sorted_importance = feature_importance[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    # Print feature importance scores
    for name, importance in zip(sorted_names, sorted_importance):
        print(f"{name}: {importance}")