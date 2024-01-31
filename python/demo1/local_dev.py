from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np

# define variables
limit = 1000000
learning_rate = 0.001
l1_reg = 0.01
l2_reg = 0 #.01
data: str = '`bt-int-ml-specialization.demo1.taxi_trips_clean`'
client = bigquery.Client(project='bt-int-ml-specialization')

"""
query_str = f"select * from {data} limit {limit}"
job = client.query(query_str)
result = job.result()
df = result.to_dataframe()
"""

# save dataframe as parquet
# df.to_parquet('python/demo1/test_data/train_clean.parquet')

df = pd.read_parquet('python/demo1/test_data/train_clean.parquet')

# check 0 entries
#df[df['fare'] > 1000]
#df[df['trip_seconds'] <= 0]
#df[df['trip_miles'] <= 0]

df_cat = pd.to_datetime(df['trip_start_timestamp']).dt.dayofweek.to_frame(name='day_of_week')
df['trip_start_datetime'] = pd.to_datetime(df['trip_start_timestamp'])
df_cat['start_month'] = df['trip_start_datetime'].dt.month
df_cat['start_day'] = df['trip_start_datetime'].dt.day
#df_cat['start_hour'] = df['trip_start_datetime'].dt.hour
#df_cat['start_minute'] = df['trip_start_datetime'].dt.minute
#df_cat['pickup_census_tract'] = df['pickup_census_tract']
#df_cat['dropoff_census_tract'] = df['dropoff_census_tract']
#df_cat['pickup_community_area'] = df['pickup_community_area']
#df_cat['dropoff_community_area'] = df['dropoff_community_area']

# fit and transform the data
df_ohe = pd.get_dummies(df_cat['start_month'], prefix='stmo')
#df_ohe = pd.get_dummies(df_cat['day_of_week'], prefix='dow')
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['start_month'], prefix='stmo')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['start_day'], prefix='stda')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['start_hour'], prefix='stho')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['start_minute'], prefix='stmi')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['pickup_census_tract'], prefix='pct')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['dropoff_census_tract'], prefix='dct')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['pickup_community_area'], prefix='pca')], axis=1)
#df_ohe = pd.concat([df_ohe, pd.get_dummies(df_cat['dropoff_community_area'], prefix='dca')], axis=1)


df_int = df[['fare', 'trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
             'dropoff_longitude']]
df_selected = pd.concat([df_int, df_ohe], axis=1)
df_selected = df_selected.astype(float)

#df_selected[['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
#             'dropoff_longitude']].fillna(df_selected[['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude',
#                                           'dropoff_latitude', 'dropoff_longitude']].mean(), inplace=True)
df_selected = df_selected.dropna() #df_selected.fillna(np.nan)
df_selected = df_selected[(df_selected[['fare', 'trip_seconds', 'trip_miles']] > 0).all(axis=1)]
df_selected = df_selected[df_selected['fare'] < 80]
df_selected = df_selected[df_selected['trip_seconds'] < 6000]
df_selected = df_selected[df_selected['trip_miles'] < 30]

# check distribution
#df_selected['fare'].hist(bins=100)
#plt.show()

#df_selected['trip_seconds'].hist(bins=100)
#plt.show()

df_selected['trip_miles'].hist(bins=100)
plt.show()

#df_selected['dropoff_longitude'].hist(bins=100)
#plt.show()

#tt = df_selected['tolls'].hist(bins=100)
#plt.show()

# too few entries with tolls
#hist, bins = np.histogram(df_selected['tolls'], bins=20)



scaler = StandardScaler()
df_selected[['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
             'dropoff_longitude']] = scaler.fit_transform(df_selected[['trip_seconds', 'trip_miles', 'pickup_latitude',
                                                                       'pickup_longitude', 'dropoff_latitude',
                                                                       'dropoff_longitude']])

# Convert pandas dataframe to numpy array
data = df_selected.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0], test_size=0.2)

# Add layers to model
# Add layers to model with L1 regularization
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(84,), kernel_regularizer=tf.keras.regularizers.l1(l1_reg)))
#model.add(tf.keras.layers.Dropout(0.1))
#model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(84,), kernel_regularizer=tf.keras.regularizers.l1(l1_reg)))
#model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# Compile model
model.compile(optimizer='adam', loss='mse', metrics='mse')

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# predict the output values for the validation dataset
y_pred = model.predict(X_test)

# calculate the R-squared value
r2 = r2_score(y_test, y_pred)

"""
result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0, scoring='r2', max_samples=100000)
importance = result.importances_mean

print(importance)
np.sum(importance)

sorted_idx = importance.argsort()[::-1]
labels = df_selected.columns.array[1:]
plt.figure(figsize=(10, 6))
plt.bar(labels[sorted_idx], importance[sorted_idx])
plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=12)
plt.ylabel('Feature Importance', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Sample model training and evaluation
original_score =  r2_score(y_test, y_pred)

labels = df_selected.columns.array[1:]

# Create a DataFrame from the 2D NumPy array and labels
X_test_df = pd.DataFrame(X_test, columns=labels)

# Function to shuffle one-hot encoded columns by prefix
def shuffle_columns_by_prefix(data, prefix):
    cols = [col for col in data.columns if col.startswith(prefix)]
    data_copy = data.copy()
    shuffled_values = np.random.permutation(data_copy[cols].values)
    data_copy[cols] = shuffled_values
    return data_copy

# List of prefixes for one-hot encoded variables
prefixes = ['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
            'dropoff_longitude', 'dow', 'stmo', 'stda', 'stho', 'stmi']
num_repetitions = 5  # Number of times to repeat the computation
subset_fraction = 0.1  # Fraction of data to use for computation

# Dictionary to store accumulated importance results
accumulated_importances = {prefix: 0 for prefix in prefixes}

# Repeat computation and accumulate importances
for i in range(num_repetitions):
    # Sample a subset of the data without replacement
    X_test_subset = X_test_df.sample(frac=subset_fraction, replace=False)
    y_test_subset = y_test[X_test_subset.index]

    for prefix in prefixes:
        print('repetition number:', i, 'prefix:', prefix)
        shuffled_data = shuffle_columns_by_prefix(X_test_subset, prefix)
        shuffled_score = r2_score(y_test_subset, model.predict(shuffled_data))
        importance = original_score - shuffled_score
        accumulated_importances[prefix] += importance

# Calculate the mean importance for each feature group
mean_feature_importance = {prefix: total_importance / num_repetitions
                           for prefix, total_importance in accumulated_importances.items()}

# Printing the mean feature importance
print("Mean Feature Importance:")
for feature, importance in mean_feature_importance.items():
    print(f"{feature}: {importance}")

# Assuming mean_feature_importance is your dictionary with feature importances
# Sort the dictionary by importance in descending order
sorted_feature_importance = dict(sorted(mean_feature_importance.items(), key=lambda item: item[1], reverse=True))

# Plotting the mean feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_feature_importance)), list(sorted_feature_importance.values()), align='center')
plt.xticks(range(len(sorted_feature_importance)), list(sorted_feature_importance.keys()), rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Mean Importance')
plt.title('Mean Feature Importance')
plt.tight_layout()  # Adjust layout to fit labels
plt.savefig('feature_importance_1.png')
plt.show()