import pandas as pd
import tensorflow as tf
import hypertune
from google.cloud import bigquery
from python.demo1.main import (define_datasets, define_model_vars, load_raw_data, build_model, compile_model,
                               save_model, fit_model)

# define variables
limit = 100000
gcp_input_table_train: str = 'bt-int-ml-specialization.demo1.taxi_trips_model_input_train'
gcp_input_table_eval: str = 'bt-int-ml-specialization.demo1.taxi_trips_model_input_eval'
client = bigquery.Client(project='bt-int-ml-specialization')
parquet_file_path_train = './python/demo1/test_data/train_parquet_file.parquet'
parquet_file_path_eval = './python/demo1/test_data/eval_parquet_file.parquet'
parquet_file_path_pred = './python/demo1/test_data/pred_parquet_file.parquet'
save_model_path = './python/demo1/test_data/saved_model.keras'
tft_record_path = ('./python/demo1/test_data/TFRecords_run_2024-01-20T15_52_24.728093-00000-of-00001'
                   '.tfrecord')

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
iodataset_train = load_raw_data(tft_record_path)
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


fit_model(model, iodataset_train_proc, epochs)
save_model(model, save_model_path)

check_df = pd.read_parquet(parquet_file_path_pred)
check_pred = check_df[['label', 'prediction']]

check_model = tf.keras.models.load_model(save_model_path)
check_model.summary()
