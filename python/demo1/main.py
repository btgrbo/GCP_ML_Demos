import hypertune
import os
import tensorflow as tf
import tensorflow_io as tfio
from fire import Fire


def define_model_vars():
    # define variables for model
    batch_size = 32
    epochs = 2  # 10
    optimizer = tf.keras.optimizers.Adam
    loss = 'mean_squared_error'

    return batch_size, epochs, optimizer, loss


# Transform the IODataset
def preprocess(features):
    # Extract the label
    label = features[b'label']

    # Extract all features from the OrderedDict
    processed_features = {key: features[key] for key in features.keys() if key != b'label'}

    # dict to list
    tensors = list(processed_features.values())

    # Convert all tensors to a common data type and stack to single tensor
    tensors = [tf.cast(tensor, dtype=tf.float32) for tensor in tensors]
    stacked_tensor = tf.stack(tensors, axis=-1)

    return stacked_tensor, label


def load_raw_data(train_file_parquet, eval_file_parquet):
    iodataset_train = tfio.experimental.IODataset.from_parquet(train_file_parquet)
    iodataset_eval = tfio.experimental.IODataset.from_parquet(eval_file_parquet)

    return iodataset_train, iodataset_eval


def define_datasets(iodataset_train, iodataset_eval, batch_size, epochs):
    # map preprocessing to datasets
    iodataset_train_proc = iodataset_train.map(preprocess)
    iodataset_eval_proc = iodataset_eval.map(preprocess)

    # Shuffle and batch the dataset
    iodataset_train_proc = iodataset_train_proc.shuffle(buffer_size=batch_size * 10).batch(batch_size)
    iodataset_eval_proc = iodataset_eval_proc.shuffle(buffer_size=batch_size * 10).batch(batch_size)

    # Repeat for the specified number of epochs
    iodataset_train_proc = iodataset_train_proc.repeat(epochs)
    iodataset_eval_proc = iodataset_eval_proc.repeat(epochs)

    return iodataset_train_proc, iodataset_eval_proc


def build_model():
    # Build a neural network model using TensorFlow and Keras
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
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
        train_file_parquet: str,
        eval_file_parquet: str,
        learning_rate: float,
):

    batch_size, epochs, optimizer, loss = define_model_vars()
    iodataset_train, iodataset_eval = load_raw_data(train_file_parquet, eval_file_parquet)
    iodataset_train_proc, iodataset_eval_proc = define_datasets(iodataset_train, iodataset_eval, batch_size, epochs)
    model = build_model()
    compile_model(model, optimizer, learning_rate, loss)
    history = fit_model(model, iodataset_train_proc, epochs, iodataset_eval_proc)
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
