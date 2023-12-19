import pandas as pd
import xgboost as xgb
import pickle
# import matplotlib.pyplot as plt

from numpy import absolute
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold


# Loading the data frame into the environment
df = pd.read_csv(r"C:/Users/OliverNowakbtelligen/OneDrive - b.telligent group/Desktop/GCP ML Demo/big-query_output.csv")

# Remove the ID columns before creating a ML model
## For test and training data set!
df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Splitting the data into training and test sets
X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definition of a baseline model without hyperparameter tuning
model1 = xgb.XGBRegressor(colsample_bytree=0.9, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0)

# Define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate model
scores = cross_val_score(model1, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# Force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

# Fitting a model
model1.fit(X_train, y_train)

# Make predictions on a test row
yhat = model1.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, yhat)
print('R2 Score:', r2)

# Variable Importance
feature_importance = model1.feature_importances_
feature_names = X_train.columns

# Plotting Variable Importance
#   plt.barh(range(len(feature_importance)), feature_importance, align='center')
#   plt.yticks(range(len(feature_importance)), feature_names)
#   plt.xlabel('Feature Importance')
#   plt.title('Variable Importance Plot')
#   plt.show()

# Save model as pickle file

with open(r"C:/Users/OliverNowakbtelligen/OneDrive - b.telligent group/Desktop/GCP ML Demo/GCP ML Demos/python/demo2/mlmodels/model1.pkl", 'wb') as model_file:
    pickle.dump(model1, model_file)