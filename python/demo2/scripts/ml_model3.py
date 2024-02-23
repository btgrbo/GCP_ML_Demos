import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Loading the data frame into the environment
df = pd.read_csv(r"C:\Users\OliverNowakbtelligen\OneDrive - b.telligent group\Desktop\Tickets\GCP ML Demos\python\demo2\test\data.csv")

# Remove the ID columns before creating an ML model
df = df.drop(['User_ID', 'Product_ID'], axis=1)

# Splitting the data into training and test sets
X = df.drop('Purchase', axis=1)
y = df['Purchase']

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build a simple feedforward neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with a single neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions on the test set
yhat_nn = model.predict(X_test).flatten()

# Calculate MAE for the neural network model
mae_nn = mean_absolute_error(y_test, yhat_nn)
print('MAE for Neural Network Model:', mae_nn)

# Calculate R2 score for the neural network model
r2_nn = r2_score(y_test, yhat_nn)
print('R2 Score for Neural Network Model:', r2_nn)
