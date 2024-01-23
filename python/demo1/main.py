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
        'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
        'payment_type': tf.io.VarLenFeature(tf.float32),
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(features, keys_to_features)

    # process label
    label = parsed_features['trip_seconds'] #todo replace by correct label
    label = tf.reshape(label, [1])

    # Convert from a SparseTensor to a dense tensor
    parsed_features['payment_type'] = tf.sparse.to_dense(parsed_features['payment_type'])

    # reshape and concat tensors
    parsed_features['trip_seconds'] = tf.reshape(parsed_features['trip_seconds'], [1])
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
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
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
