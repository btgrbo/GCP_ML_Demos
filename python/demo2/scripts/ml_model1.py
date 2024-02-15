import pandas as pd
import xgboost as xgb
# import pickle
# import matplotlib.pyplot as plt

from numpy import absolute
from scipy.stats import chi2_contingency
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold


# Loading the data frame into the environment
df = pd.read_csv(r"C:\Users\OliverNowakbtelligen\OneDrive - b.telligent group\Desktop\Tickets\GCP ML Demos\python\demo2\test\data.csv")

# Remove the ID columns before creating a ML model
## For test and training data set!
df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Splitting the data into training and test sets
X = df.drop('Purchase', axis=1)
y = df['Purchase']


# 'Purchase' as your dependent variable
dependent_variable = 'Purchase'

# List to store chi-square results
chi_square_results = []

# Loop through all binary variables (excluding 'Purchase')
for column in df.columns:
    if column != dependent_variable and len(df[column].unique()) == 2:
        # Create a contingency table
        contingency_table = pd.crosstab(df[column], df[dependent_variable])
        
        # Perform chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
        
        # Append results to the list
        chi_square_results.append((column, chi2, p))

# Sort the results based on chi-square values in descending order
chi_square_results.sort(key=lambda x: x[1], reverse=True)

# Print the results
for result in chi_square_results:
    print(f"Variable: {result[0]}, Chi-square: {result[1]}, P-value: {result[2]}")


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

# with open(r"C:/Users/OliverNowakbtelligen/OneDrive - b.telligent group/Desktop/GCP ML Demo/GCP ML Demos/python/demo2/mlmodels/model1.pkl", 'wb') as model_file:
# pickle.dump(model1, model_file)


