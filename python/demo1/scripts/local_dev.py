import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import keras
import pandas as pd
import scipy.stats as stats
import pickle
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# set random seeds for random, np and tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

keras.utils.set_random_seed(812)


def get_data(dataset: str, limit: int, project: str) -> pd.DataFrame:
    client = bigquery.Client(project=project)
    query_str = f"select * from {dataset} limit {limit}"
    job = client.query(query_str)
    result = job.result()
    data_df = result.to_dataframe()

    return data_df


def save_parquet(dataset: pd.DataFrame, filename: str) -> None:
    dataset.to_parquet(f"python/demo1/test_data/{filename}.parquet")


def read_parquet(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(f"python/demo1/test_data/{filename}.parquet")

    return df


def clean_df(dataset: pd.DataFrame, cols: list[str], thresholds: dict[str, int]) -> pd.DataFrame:
    dataset = dataset[cols]
    df_selected = dataset.dropna()
    for int_var, threshold in thresholds.items():
        df_selected = df_selected[(df_selected[int_var] > 0) & (df_selected[int_var] < threshold)]

    return df_selected


def process_timestamp(dataset: pd.DataFrame, timestamp_name: str) -> pd.DataFrame:
    dataset['day_of_week'] = pd.to_datetime(dataset[timestamp_name]).dt.dayofweek
    dataset['trip_start_datetime'] = pd.to_datetime(dataset['trip_start_timestamp'])
    dataset['start_month'] = dataset['trip_start_datetime'].dt.month
    dataset['start_day'] = dataset['trip_start_datetime'].dt.day
    dataset['start_hour'] = dataset['trip_start_datetime'].dt.hour
    dataset['start_minute'] = dataset['trip_start_datetime'].dt.minute
    dataset.drop([timestamp_name, 'trip_start_datetime'], axis=1, inplace=True)

    return dataset


def apply_ohe(dataset: pd.DataFrame, cols: list[str], prefixs: list[str]) -> pd.DataFrame:
    for col, prefix in zip(cols, prefixs):
        dataset = pd.concat([dataset, pd.get_dummies(dataset[col], prefix=prefix)], axis=1)
        dataset.drop(col, axis=1, inplace=True)

    return dataset


def df_to_float(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.astype(float)

    return dataset


def plot_dist(dataset: pd.DataFrame,
              col: str,
              nbins: int,
              save_fig: bool,
              filename: str,
              x_label: str = None,
              y_label: str = None) -> None:

    # Convert dimensions from cm to inches (1 inch = 2.54 cm)
    width_in_inches = 5 / 2.54
    height_in_inches = 4 / 2.54

    # Create a figure and axis with the specified size
    fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))

    # Plot histogram
    dataset[col].hist(bins=nbins, grid=False, ax=ax)  # Turn off grid

    # Set x and y labels
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move left and bottom spines away from the axes (detaching axes)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Optionally, turn off ticks where there is no spine
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()

    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{filename}.png")
    plt.show()


def scale_ints(dataset: pd.DataFrame, int_vars: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    dataset[int_vars] = scaler.fit_transform(dataset[int_vars])

    return dataset


def split_data(dataset: pd.DataFrame, test_size: float) -> dict[str, np.ndarray]:
    # Convert pandas dataframe to numpy array
    data = dataset.values
    # Split data into training and testing sets
    data_split = {'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None}

    (data_split['X_train'], data_split['X_test'],
     data_split['y_train'], data_split['y_test']) = train_test_split(data[:, 1:], data[:, 0],
                                                                     test_size=test_size, random_state=0)

    return data_split


def train_model(hidden_units: int,
                data_split: dict[str, np.ndarray],
                epochs: int,
                batch_size: int) -> tuple[tf.keras.callbacks.History, tf.keras.models.Model]:

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=10,  # Number of epochs to wait after min has been hit
        mode='min',  # Minimizing the monitored quantity ('val_loss' in this case)
        verbose=1,
        restore_best_weights=True  # Restores model weights from the epoch with the minimum monitored quantity
    )

    # build model
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.01))  # Dropout layer after the input layer

    # Hidden layers
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.01))  # Dropout layer
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.01))  # Dropout layer

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics='mse')

    # Train model
    history = model.fit(data_split['X_train'], data_split['y_train'], epochs=epochs, batch_size=batch_size,
                        validation_data=(data_split['X_test'], data_split['y_test']), shuffle=True,
                        callbacks=early_stopping)

    return history, model



def plot_training(history: tf.keras.callbacks.History,
                  save_fig: bool,
                  filename: str,
                  min_epoch: int) -> None:
    # Convert dimensions from cm to inches
    width_in_inches = 10 / 2.54
    height_in_inches = 8 / 2.54

    # Hardcoded xlim and ylim values
    xlim_values = (1, 60)  # Ensuring min_epoch is within the xlim
    ylim_values = (1, 9)

    # Create a figure with the specified size
    plt.figure(figsize=(width_in_inches, height_in_inches))

    # Adjusted x-values to range from 1 to the length of the history loss array + 1
    x_values = np.arange(1, len(history.history['loss']) + 1)

    # Plot Training and Validation Loss with adjusted x-values
    plt.plot(x_values, history.history['loss'], label='Training Loss')
    plt.plot(x_values, history.history['val_loss'], label='Validation Loss')

    # Add a dashed vertical line at min_epoch
    plt.axvline(x=min_epoch, color='k', linestyle='--', label=f'Min Epoch ({min_epoch})')

    # Set title, labels, and axes limits
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(xlim_values)
    plt.ylim(ylim_values)

    # Hide top and right spines and adjust bottom and left spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_position(('outward', 10))
    plt.gca().spines['left'].set_position(('outward', 10))

    # Set specific tick labels for x and y axes (adjust as necessary)
    plt.yticks([1, 5, 9])
    plt.xticks([1, 20, 60])

    # Adjust tick parameters and add legend
    plt.tick_params(top=False, labeltop=False, right=False, labelright=False)
    #plt.legend()

    # Use tight layout to adjust subplot parameters
    plt.tight_layout()

    # Save figure if required
    if save_fig and filename:
        plt.savefig(f"{filename}.png")

    # Show the plot
    plt.show()


def get_r2(model: tf.keras.models.Model, split_data: dict) -> float:
    # predict the output values for the validation dataset
    y_pred = model.predict(split_data['X_test'])

    # calculate the R-squared value
    r2 = r2_score(split_data['y_test'], y_pred)
    ### model: 0.9686729986792358
    return r2


# Function to shuffle one-hot encoded columns by prefix
def shuffle_columns_by_prefix(data, prefix):
    cols = [col for col in data.columns if col.startswith(prefix)]
    data_copy = data.copy()
    shuffled_values = np.random.default_rng(seed=42).permutation(data_copy[cols].values)
    data_copy[cols] = shuffled_values

    return data_copy


def compute_feat_importance(orig_r2: float,
                            labels: list[str],
                            split_data: dict[str, np.ndarray],
                            prefixes: list[str],
                            num_repetitions: int,
                            subset_fraction: float,
                            model: tf.keras.models.Model) -> tuple[dict[str, float], dict[str, tuple]]:

    # Create a DataFrame from the 2D NumPy array and labels
    X_test_df = pd.DataFrame(split_data['X_test'], columns=labels)

    # Store importances for each feature for each repetition
    feature_importances = {prefix: [] for prefix in prefixes}

    for i in range(num_repetitions):
        # Sample a subset of the data without replacement
        X_test_subset = X_test_df.sample(frac=subset_fraction, replace=False, random_state=i)
        y_test_subset = split_data['y_test'][X_test_subset.index]

        for prefix in prefixes:
            shuffled_data = shuffle_columns_by_prefix(X_test_subset, prefix)
            shuffled_score = r2_score(y_test_subset, model.predict(shuffled_data))
            importance = orig_r2 - shuffled_score
            feature_importances[prefix].append(importance)

    # Compute mean and 95% confidence interval for each feature group
    mean_feature_importance = {}
    confidence_intervals = {}


    for prefix, importances in feature_importances.items():
        mean_importance = np.mean(importances)
        mean_feature_importance[prefix] = mean_importance
        # Compute 95% confidence interval
        conf_int = stats.t.interval(0.95, len(importances)-1, loc=mean_importance, scale=stats.sem(importances))
        confidence_intervals[prefix] = conf_int

    return mean_feature_importance, confidence_intervals


def plot_feature_importance(mean_feature_importance: dict[str, float],
                            confidence_intervals: dict[str, tuple],
                            save_fig: bool = False,
                            filename: str = None) -> None:
    # Convert dimensions from cm to inches (1 inch = 2.54 cm)
    width_in_inches = 16 / 2.54  # 16 cm wide
    height_in_inches = 8 / 2.54  # 8 cm high

    # Plotting with updated figure size
    plt.figure(figsize=(width_in_inches, height_in_inches))

    # Sort features by mean importance
    sorted_features = sorted(mean_feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, means = zip(*sorted_features)

    # Compute error from mean to the upper bound of the confidence interval and determine colors
    errors = [confidence_intervals[feature][1] - mean for feature, mean in zip(features, means)]
    colors = ['red' if (confidence_intervals[feature][0] <= 0 <= confidence_intervals[feature][1]) else 'blue' for
              feature in features]

    # Plot each point individually to assign colors
    for feature, mean, error, color in zip(features, means, errors, colors):
        plt.errorbar(feature, mean, yerr=[[error], [0]], fmt='o', color=color, ecolor='lightgray', elinewidth=3,
                     capsize=0)

    plt.xlabel('Features')
    plt.ylabel('Mean Importance')
    plt.title('Feature Importance')  # Update title

    # Hide top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Move left and bottom spines away from the axes (detaching axes)
    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().yaxis.tick_left()
    plt.gca().xaxis.tick_bottom()

    plt.xticks(rotation=90)  # Adjust xticklabels to be arranged at a 45-degree angle

    plt.tight_layout()

    # Save figure if required
    if save_fig and filename:
        plt.savefig(f"{filename}.png", dpi=300)  # dpi can be adjusted for higher resolution

    plt.show()


def eval_variance(model:tf.keras.models.Model, X, y, cv:int = 10) -> None:
    # Assuming X, y are your features and target variable
    # Note: Setting scoring to 'neg_mean_squared_error' because cross_val_score returns negative values for MSE
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')

    mse_scores = -scores  # Convert to positive MSE scores
    print(f"MSE scores across folds: {mse_scores}")
    print(f"Variance of MSE across folds: {np.var(mse_scores)}")


def eval_bias(y_true, y_pred)-> None:
    # Assuming y_true are the actual values and y_pred are the model predictions
    errors = y_true - y_pred
    mean_error = np.mean(errors)

    print(f"Mean Error (Bias): {mean_error}")


# define variables
limit: int = 1000000
epochs: int = 100
batch_size: int = 32
learning_rate: float = 0.001
test_size: float = 0.2
data_w_outlier: str = '`bt-int-ml-specialization.demo1.taxi_trips_clean`'
data_wo_outlier: str = '`bt-int-ml-specialization.demo1.taxi_trips_ex_outlier`'
project: str = 'bt-int-ml-specialization'
hidden_units: int = 16
timestamp_raw = 'trip_start_timestamp'
pred_nom = ['start_month', 'day_of_week', 'start_day', 'start_hour', 'start_minute']
prefix_pred = ['stmo', 'dow', 'stda', 'stho', 'stmi']
target = ['fare']
pred_int = ['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
            'dropoff_longitude']
all_cols = target + pred_int + ['trip_start_timestamp']
thresholds_input = {'fare': 80,
                    'trip_seconds': 6000,
                    'trip_miles': 30}

# List of prefixes for one-hot encoded variables
prefixes_input = ['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                  'dropoff_longitude', 'dow', 'stmo', 'stda', 'stho', 'stmi']
num_repetitions_input = 100  # Number of times to repeat the computation
subset_fraction_input = 0.2  # Fraction of data to use for computation



# plot histograms for original, cleaned, and normalized data
#df_clean = get_data(dataset=data_w_outlier, limit=limit, project=project)
#save_parquet(dataset=df_clean, filename='train_clean')
# before cleaning
df = read_parquet(filename='train_clean')
plot_dist(dataset=df, col='trip_miles', nbins=100, save_fig=True, filename='dist_raw', x_label='Trip miles',
          y_label='# occurences')
# after cleaning
clean_df = clean_df(dataset=df, cols=all_cols, thresholds=thresholds_input)
plot_dist(dataset=clean_df, col='trip_miles', nbins=100, save_fig=True, filename='dist_cleaned', x_label='Trip miles')
# after scaling
scaled_df = scale_ints(dataset=clean_df, int_vars=pred_int)
plot_dist(dataset=scaled_df, col='trip_miles', nbins=100, save_fig=True, filename='dist_scaled',
          x_label='Trip miles [norm.]')

# train model
#df_ex_outlier = get_data(dataset=data_wo_outlier, limit=limit, project=project)
#save_parquet(dataset=df_ex_outlier, filename='train_ex_outlier')
df_ex_outlier = read_parquet(filename='train_ex_outlier')
scaled_df = scale_ints(dataset=df_ex_outlier, int_vars=pred_int)
df_time = process_timestamp(dataset=scaled_df, timestamp_name=timestamp_raw)
df_ohe = apply_ohe(dataset=df_time, cols=pred_nom, prefixs=prefix_pred)
df_float = df_to_float(dataset=df_ohe)
data_split = split_data(dataset=df_float, test_size=test_size)
train_hist, trained_model = train_model(hidden_units=hidden_units, data_split=data_split, epochs=epochs,
                                        batch_size=batch_size)
trained_model.save('./trained_model.h5')
with open('./train_hist.pkl', 'wb') as file:
    pickle.dump(train_hist.history, file)
plot_training(history=train_hist, save_fig=True, filename='train_hist', min_epoch=49)
r2 = get_r2(model=trained_model, split_data=data_split)
labels = df_float.columns.array[1:]
mean_feature_importance, confidence_intervals = compute_feat_importance(orig_r2=r2,
                                                                        labels=labels,
                                                                        split_data=data_split,
                                                                        prefixes=prefixes_input,
                                                                        num_repetitions=num_repetitions_input,
                                                                        subset_fraction=subset_fraction_input,
                                                                        model=trained_model)
with open('./mean_feature_importance.pkl', 'wb') as file:
    pickle.dump(mean_feature_importance, file)
with open('./confidence_intervals.pkl', 'wb') as file:
    pickle.dump(confidence_intervals, file)
plot_feature_importance(mean_feature_importance=mean_feature_importance,
                        confidence_intervals=confidence_intervals,
                        save_fig=True,
                        filename='feature_importance')


### retrain without stmi
pred_nom = ['start_month', 'day_of_week', 'start_day', 'start_hour']
prefix_pred = ['stmo', 'dow', 'stda', 'stho']
prefixes_input = ['trip_seconds', 'trip_miles', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
                  'dropoff_longitude', 'dow', 'stmo', 'stda', 'stho']
df_ex_outlier = read_parquet(filename='train_ex_outlier')
scaled_df = scale_ints(dataset=df_ex_outlier, int_vars=pred_int)
df_time = process_timestamp(dataset=scaled_df, timestamp_name=timestamp_raw)
df_time.drop(['start_minute'], axis=1, inplace=True)
df_ohe = apply_ohe(dataset=df_time, cols=pred_nom, prefixs=prefix_pred)
df_float = df_to_float(dataset=df_ohe)
data_split = split_data(dataset=df_float, test_size=test_size)
train_hist_imp, trained_model_imp = train_model(hidden_units=hidden_units, data_split=data_split, epochs=epochs,
                                                batch_size=batch_size)
trained_model_imp.save('./trained_model_imp.h5')
with open('./train_hist_imp.pkl', 'wb') as file:
    pickle.dump(train_hist_imp.history, file)
plot_training(history=train_hist_imp, save_fig=True, filename='train_hist_imp', min_epoch=25)
r2_imp = get_r2(model=trained_model_imp, split_data=data_split)

tf.keras.utils.plot_model(trained_model_imp,
                          to_file='./model_architecture.png',
                          show_shapes=True,
                          show_layer_names=True,
                          dpi=300)

