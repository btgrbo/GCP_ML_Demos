from google.cloud import storage
import json
import os
import tensorflow as tf

feature_description: dict = {
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


def _parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def tfrecord_to_jsonl_gcs(tfrecord_gcs_path, jsonl_gcs_path):
    # Initialize GCS client
    storage_client = storage.Client()
    # Temporary paths for downloading and uploading
    temp_tfrecord_path = '/tmp/temp_file.tfrecord'
    temp_jsonl_path = '/tmp/temp_file.jsonl'

    # Download the TFRecord file from GCS
    # Assuming tfrecord_gcs_path is like 'gs://bucket_name/path/to/file.tfrecord'
    bucket_name = tfrecord_gcs_path.split('/')[2]
    blob_name = '/'.join(tfrecord_gcs_path.split('/')[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(temp_tfrecord_path)

    with open(temp_jsonl_path, 'w') as jsonl_file:
        for raw_record in tf.data.TFRecordDataset(temp_tfrecord_path):
            example = _parse_example(raw_record)
            # Convert VarLenFeature to dense and decode bytes to string or float
            processed_example = {key: value.numpy().tolist() if hasattr(value, 'numpy') else value for key, value in
                                 example.items()}

            # Extract fare and concatenate other features into dense_input
            dense_input = []
            for key, value in processed_example.items():
                if key != 'fare':
                    if isinstance(value, list):
                        dense_input.extend(value)
                    if isinstance(value, tf.SparseTensor):
                        dense_input.extend(value._numpy().tolist())
                    else:  # For single float values not in a list
                        dense_input.append(value)

            jsonl_line = json.dumps({"dense_input": dense_input, "fare": processed_example['fare']})
            jsonl_file.write(jsonl_line + '\n')

    # Upload the JSONL file to GCS
    jsonl_bucket_name = jsonl_gcs_path.split('/')[2]
    jsonl_blob_name = '/'.join(jsonl_gcs_path.split('/')[3:])
    jsonl_bucket = storage_client.bucket(jsonl_bucket_name)
    jsonl_blob = jsonl_bucket.blob(jsonl_blob_name)
    jsonl_blob.upload_from_filename(temp_jsonl_path)

    # Clean up temporary files
    os.remove(temp_tfrecord_path)
    os.remove(temp_jsonl_path)


# Example usage
tfrecord_gcs_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/demo1-2024-02-22-17-42-11-00000-of-00001.tfrecord'
jsonl_gcs_path = 'gs://bt-int-ml-specialization-ml-demo1/eval_results/test_data.jsonl'
tfrecord_to_jsonl_gcs(tfrecord_gcs_path, jsonl_gcs_path)
