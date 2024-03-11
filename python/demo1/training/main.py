import os

import hypertune
import tensorflow as tf
from fire import Fire


def define_model_vars() -> tuple[int, int, tf.keras.optimizers, str]:
    # define variables for model
    batch_size = 32
    epochs = 7
    optimizer = tf.keras.optimizers.Adam
    loss = 'mean_squared_error'

    return batch_size, epochs, optimizer, loss


def preprocess(features: tf.data.TFRecordDataset) -> tuple[tf.Tensor, tf.Tensor]:

    # Define parsing schema
    keys_to_features = {
        'fare': tf.io.FixedLenFeature([], tf.float32),
        'trip_miles': tf.io.FixedLenFeature([], tf.float32),
        'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
        'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
        'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
        'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
        'pickup_longitude': tf.io.FixedLenFeature([], tf.float32),
        'start_hour': tf.io.VarLenFeature(tf.float32),
        'start_month': tf.io.VarLenFeature(tf.float32),
        'start_date': tf.io.VarLenFeature(tf.float32),
        'day_of_week': tf.io.VarLenFeature(tf.float32)
    }

    # Load one example
    parsed_features = tf.io.parse_example(features, keys_to_features)

    # process label
    label = tf.expand_dims(parsed_features["fare"], axis=-1)
    del parsed_features['fare']

    # Convert from a SparseTensor to a dense tensor
    ohe_vars = ['start_month', 'start_date', 'day_of_week', 'start_hour']
    for ohe_var in ohe_vars:
        parsed_features[ohe_var] = tf.sparse.to_dense(parsed_features[ohe_var])

    flt_vars = ['trip_miles', 'trip_seconds', 'dropoff_longitude', 'dropoff_latitude', 'pickup_longitude',
                'pickup_latitude']
    for flt_var in flt_vars:
        # reshape and concat tensors
        parsed_features[flt_var] = tf.expand_dims(parsed_features[flt_var], axis=-1)

    tensors = tf.concat(list(parsed_features.values()), axis=1)

    return tensors, label


def load_raw_data(tft_record_path: str) -> tf.data.TFRecordDataset:

    path_w_suffix = tft_record_path + "*.tfrecord"
    file_names = tf.io.gfile.glob(path_w_suffix)
    iodataset_train = tf.data.TFRecordDataset(file_names)

    return iodataset_train


def define_datasets(iodataset_train: tf.data.TFRecordDataset,
                    batch_size: int,
                    epochs: int) -> tf.data.TFRecordDataset:

    # Shuffle and batch the dataset
    iodataset_train_proc = iodataset_train.shuffle(buffer_size=batch_size * 10).batch(batch_size).repeat(epochs)

    # map preprocessing to datasets
    iodataset_train_proc = iodataset_train_proc.map(preprocess)

    return iodataset_train_proc


def build_model(dropout_rate: float) -> tf.keras.models.Sequential:

    # build model
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(80,)))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer after the input layer

    # Hidden layers
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    return model


def compile_model(model: tf.keras.models.Sequential,
                  optimizer: tf.keras.optimizers,
                  learning_rate: float,
                  loss: str) -> None:
    # Compile the model
    model.compile(optimizer=optimizer(learning_rate), loss=loss)


def fit_model(model: tf.keras.models.Sequential,
              iodataset_train_proc: tf.data.TFRecordDataset,
              epochs: int,
              iodataset_eval_proc: tf.data.TFRecordDataset) -> tf.keras.callbacks.History:

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=3,  # Number of epochs to wait after min has been hit
        mode="min",  # Minimizing the monitored quantity ('val_loss' in this case)
        verbose=1,
        restore_best_weights=True,  # Restores model weights from the epoch with the minimum monitored quantity
    )

    # fit model
    history = model.fit(iodataset_train_proc, epochs=epochs, validation_data=iodataset_eval_proc,
                        callbacks=early_stopping)

    return history


def save_model(model: tf.keras.models.Sequential, model_file: str) -> None:
    model.save(model_file)
    print(f"Model saved to {model_file}")


def main(
        train_file_path: str,
        learning_rate: float,
        dropout_rate: float
):
    batch_size, epochs, optimizer, loss = define_model_vars()
    iodataset_train = load_raw_data(tft_record_path=train_file_path)
    iodataset_train_proc = define_datasets(iodataset_train=iodataset_train,
                                           batch_size=batch_size,
                                           epochs=epochs)
    model = build_model(dropout_rate=dropout_rate)
    compile_model(model=model,
                  optimizer=optimizer,
                  learning_rate=learning_rate,
                  loss=loss)import hypertune
import os
import tensorflow as tf
from fire import Fire


def define_model_vars() -> tuple[int, int, tf.keras.optimizers, str]:
    # define variables for model
    batch_size = 32
    epochs = 7
    optimizer = tf.keras.optimizers.Adam
    loss = 'mean_squared_error'

    return batch_size, epochs, optimizer, loss


def preprocess(features: tf.data.TFRecordDataset) -> tuple[tf.Tensor, tf.Tensor]:

    # Define parsing schema
    keys_to_features = {
        'fare': tf.io.FixedLenFeature([], tf.float32),
        'trip_miles': tf.io.FixedLenFeature([], tf.float32),
        'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
        'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
        'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
        'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
        'pickup_longitude': tf.io.FixedLenFeature([], tf.float32),
        'start_hour': tf.io.VarLenFeature(tf.float32),
        'start_month': tf.io.VarLenFeature(tf.float32),
        'start_date': tf.io.VarLenFeature(tf.float32),
        'day_of_week': tf.io.VarLenFeature(tf.float32)
    }

    # Load one example
    parsed_features = tf.io.parse_example(features, keys_to_features)

    # process label
    label = tf.expand_dims(parsed_features['fare'], axis=-1)
    del parsed_features['fare']

    # Convert from a SparseTensor to a dense tensor
    ohe_vars = ['start_month', 'start_date', 'day_of_week', 'start_hour']
    for ohe_var in ohe_vars:
        parsed_features[ohe_var] = tf.sparse.to_dense(parsed_features[ohe_var])

    flt_vars = ['trip_miles', 'trip_seconds', 'dropoff_longitude', 'dropoff_latitude', 'pickup_longitude',
                'pickup_latitude']
    for flt_var in flt_vars:
        # reshape and concat tensors
        parsed_features[flt_var] = tf.expand_dims(parsed_features[flt_var], axis=-1)

    tensors = tf.concat(list(parsed_features.values()), axis=1)

    return tensors, label


def load_raw_data(tft_record_path: str) -> tf.data.TFRecordDataset:

    path_w_suffix = tft_record_path + '*.tfrecord'
    file_names = tf.io.gfile.glob(path_w_suffix)
    iodataset_train = tf.data.TFRecordDataset(file_names)

    return iodataset_train


def define_datasets(iodataset_train: tf.data.TFRecordDataset,
                    batch_size: int,
                    epochs: int) -> tf.data.TFRecordDataset:

    # Shuffle and batch the dataset
    iodataset_train_proc = iodataset_train.shuffle(buffer_size=batch_size * 10).batch(batch_size).repeat(epochs)

    # map preprocessing to datasets
    iodataset_train_proc = iodataset_train_proc.map(preprocess)

    return iodataset_train_proc


def build_model(dropout_rate: float) -> tf.keras.models.Sequential:

    # build model
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(80,)))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer after the input layer

    # Hidden layers
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))  # Dropout layer

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    return model


def compile_model(model: tf.keras.models.Sequential,
                  optimizer: tf.keras.optimizers,
                  learning_rate: float,
                  loss: str) -> None:
    # Compile the model
    model.compile(optimizer=optimizer(learning_rate), loss=loss)


def fit_model(model: tf.keras.models.Sequential,
              iodataset_train_proc: tf.data.TFRecordDataset,
              epochs: int,
              iodataset_eval_proc: tf.data.TFRecordDataset) -> tf.keras.callbacks.History:

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=3,  # Number of epochs to wait after min has been hit
        mode='min',  # Minimizing the monitored quantity ('val_loss' in this case)
        verbose=1,
        restore_best_weights=True  # Restores model weights from the epoch with the minimum monitored quantity
    )

    # fit model
    history = model.fit(iodataset_train_proc, epochs=epochs, validation_data=iodataset_eval_proc,
                        callbacks=early_stopping)

    return history


def save_model(model: tf.keras.models.Sequential, model_file: str) -> None:
    model.save(model_file)
    print(f"Model saved to {model_file}")


def main(
        train_file_path: str,
        learning_rate: float,
        dropout_rate: float
):
    batch_size, epochs, optimizer, loss = define_model_vars()
    iodataset_train = load_raw_data(tft_record_path=train_file_path)
    iodataset_train_proc = define_datasets(iodataset_train=iodataset_train,
                                           batch_size=batch_size,
                                           epochs=epochs)
    model = build_model(dropout_rate=dropout_rate)
    compile_model(model=model,
                  optimizer=optimizer,
                  learning_rate=learning_rate,
                  loss=loss)

    len_dataset = len(list(iodataset_train_proc))

    history = fit_model(model=model,
                        iodataset_train_proc=iodataset_train_proc.take(int(len_dataset*0.8)),
                        epochs=epochs,
                        iodataset_eval_proc=iodataset_train_proc.skip(int(len_dataset*0.8)))

    hp_metric = history.history['val_loss'][-1]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='loss',
        metric_value=hp_metric,
        global_step=epochs
    )
    model_dir = os.getenv('AIP_MODEL_DIR')
    save_model(model, model_dir)


if __name__ == '__main__':
    # run with `python demo2/main.py \
    #   --train_file_parquet demo2/train.parquet \
    #   --eval_file_parquet demo2/eval.parquet \
    #   --model_file demo2/model.xgb \
    #   --eval_output_file_parquet demo2/eval_with_predictions.parquet`

    Fire(main)

    len_dataset = 4000740

    history = fit_model(
        model=model,
        iodataset_train_proc=iodataset_train_proc.take(int(len_dataset * 0.8)),
        epochs=epochs,
        iodataset_eval_proc=iodataset_train_proc.skip(int(len_dataset * 0.8)),
    )

    hp_metric = history.history['val_loss'][-1]
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='loss',
        metric_value=hp_metric,
        global_step=epochs
    )
    model_dir = os.getenv('AIP_MODEL_DIR')
    save_model(model, model_dir)


if __name__ == '__main__':
    # run with `python demo2/main.py \
    #   --train_file_parquet demo2/train.parquet \
    #   --eval_file_parquet demo2/eval.parquet \
    #   --model_file demo2/model.xgb \
    #   --eval_output_file_parquet demo2/eval_with_predictions.parquet`

    Fire(main)
