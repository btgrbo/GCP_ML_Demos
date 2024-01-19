import pandas as pd
import tensorflow as tf
import hypertune
from google.cloud import bigquery
from main import (define_datasets, define_model_vars, load_raw_data, build_model, compile_model, save_model, fit_model,
                  export_predictions)

# define variables
limit = 100000
gcp_input_table_train: str = 'bt-int-ml-specialization.demo1.taxi_trips_model_input_train'
gcp_input_table_eval: str = 'bt-int-ml-specialization.demo1.taxi_trips_model_input_eval'
client = bigquery.Client(project='bt-int-ml-specialization')
parquet_file_path_train = './GCP%20ML%20Demos/python/demo1/test_data/train_parquet_file.parquet'
parquet_file_path_eval = './GCP%20ML%20Demos/python/demo1/test_data/eval_parquet_file.parquet'
parquet_file_path_pred = './GCP%20ML%20Demos/python/demo1/test_data/pred_parquet_file.parquet'
save_model_path = './GCP%20ML%20Demos/python/demo1/test_data/saved_model.keras'

# save training data as parquet
query_str = f"select * from {gcp_input_table_train} limit {limit}"
job = client.query(query_str)
result = job.result()
df = result.to_dataframe()
df.to_parquet(parquet_file_path_train)

# save evaluation data as parquet
query_str = f"select * from {gcp_input_table_eval} limit {limit}"
job = client.query(query_str)
result = job.result()
df = result.to_dataframe()
df.to_parquet(parquet_file_path_eval)

learning_rate = 0.0001

batch_size, epochs, optimizer, loss = define_model_vars()
iodataset_train, iodataset_eval = load_raw_data(parquet_file_path_train, parquet_file_path_eval)
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


fit_model(model, iodataset_train_proc, epochs, iodataset_eval_proc)
save_model(model, save_model_path)
export_predictions(iodataset_eval, batch_size, model, parquet_file_path_pred)

check_df = pd.read_parquet(parquet_file_path_pred)
check_pred = check_df[['label', 'prediction']]

check_model = tf.keras.models.load_model(save_model_path)
check_model.summary()

