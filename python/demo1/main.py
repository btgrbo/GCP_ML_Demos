import hypertune
import os
import tensorflow as tf
from fire import Fire


def define_model_vars():
    # define variables for model
    batch_size = 32
    epochs = 2  # 10
    optimizer = tf.keras.optimizers.Adam
    loss = 'mean_squared_error'

    return batch_size, epochs, optimizer, loss


def preprocess(features):

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
    parsed_features = tf.io.parse_single_example(features, keys_to_features)

    # process label
    label = parsed_features['fare']
    label = tf.reshape(label, [1])
    del parsed_features['fare']

    # Convert from a SparseTensor to a dense tensor
    ohe_vars = ['start_month', 'start_date', 'day_of_week']
    for ohe_var in ohe_vars:
        parsed_features[ohe_var] = tf.sparse.to_dense(parsed_features[ohe_var])

    flt_vars = ['trip_miles', 'trip_seconds', 'dropoff_longitude', 'dropoff_latitude', 'pickup_longitude',
                'pickup_latitude']
    for flt_var in flt_vars:
        # reshape and concat tensors
        parsed_features[flt_var] = tf.reshape(parsed_features[flt_var], [1])

    tensors = list(parsed_features.values())
    tensors = tf.concat(tensors, axis=-1)
    tensors = tf.reshape(tensors, [-1,])

    return tensors, label


def load_raw_data(tft_record_path):

    iodataset_train = tf.data.TFRecordDataset(tft_record_path)

    return iodataset_train


def define_datasets(iodataset_train, batch_size, epochs):

    # map preprocessing to datasets
    iodataset_train_proc = iodataset_train.map(preprocess)

    # Shuffle and batch the dataset
    iodataset_train_proc = iodataset_train_proc.shuffle(buffer_size=batch_size * 10).batch(batch_size)

    # Repeat for the specified number of epochs
    iodataset_train_proc = iodataset_train_proc.repeat(epochs)

    return iodataset_train_proc


def build_model():

    # Build a neural network model using TensorFlow and Keras
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(80,)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    return model


def compile_model(model, optimizer, learning_rate, loss):
    # Compile the model
    model.compile(optimizer=optimizer(learning_rate), loss=loss)


def fit_model(model, iodataset_train_proc, epochs, iodataset_eval_proc):
    # fit model
    history = model.fit(iodataset_train_proc, epochs=epochs, validation_data=iodataset_eval_proc)

    return history


def save_model(model, model_file):
    model.save(model_file)
    print(f"Model saved to {model_file}")


def main(
        train_file_path: str,
        learning_rate: float,
):

    batch_size, epochs, optimizer, loss = define_model_vars()
    iodataset_train = load_raw_data(train_file_path)
    iodataset_train_proc = define_datasets(iodataset_train, batch_size, epochs)
    model = build_model()
    compile_model(model, optimizer, learning_rate, loss)
    history = fit_model(model, iodataset_train_proc.take(int(batch_size*0.8)), epochs,
                        iodataset_train_proc.skip(int(batch_size*0.8)))
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
