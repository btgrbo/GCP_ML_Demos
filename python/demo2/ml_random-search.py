import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score
from numpy import absolute
import scipy.stats

# Loading the data frame into the environment
df = pd.read_csv(r"C:/Users/OliverNowakbtelligen/OneDrive - b.telligent group/Desktop/GCP ML Demo/big-query_output.csv")

# Remove the ID columns before creating an ML model
df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Splitting the data into training and test sets
X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter distributions for hyperparameter tuning
param_dist = {
    'n_estimators': scipy.stats.randint(100, 300),
    'max_depth': scipy.stats.randint(3, 6),
    'learning_rate': scipy.stats.uniform(0.01, 0.2),
    'subsample': scipy.stats.uniform(0.8, 0.2),
    'colsample_bytree': scipy.stats.uniform(0.8, 0.2)
}

# Create an XGBoost regressor model
base_model = xgb.XGBRegressor()

# Define the randomized search with cross-validation
random_search = RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=10, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, random_state=42, verbose=2)

# Perform the randomized search on the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters from the random search
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model from the random search
best_model = random_search.best_estimator_

# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
cv_scores = absolute(cv_scores)
print('Mean MAE with Best Model: %.3f (%.3f)' % (cv_scores.mean(), cv_scores.std()))

# Fit the best model to the training data
best_model.fit(X_train, y_train)

# Make predictions on a test row
yhat = best_model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, yhat)
print('R2 Score:', r2)


# Best Hyperparameters: {'colsample_bytree': 0.8285733635843882, 'learning_rate': 0.14017769458977059, 'max_depth': 3, 'n_estimators': 101, 'subsample': 0.944399754453365}
# Mean MAE with Best Model: 2432.756 (111.908)
# R2 Score = 0.630