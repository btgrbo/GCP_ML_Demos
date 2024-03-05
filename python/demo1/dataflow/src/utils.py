from typing import Any

import pytz
import numpy as np
import tensorflow as tf
import json
from apache_beam import Row
from apache_beam.ml import MLTransform
from apache_beam.ml.transforms.tft import TFTOperation
from datetime import datetime
from google.cloud import aiplatform
from itertools import chain

feature_description_flt: dict = {
    'fare': tf.io.FixedLenFeature([], tf.float32),
    'trip_miles': tf.io.FixedLenFeature([], tf.float32),
    'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
    'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
    'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
    'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
    'pickup_longitude': tf.io.FixedLenFeature([], tf.float32)
}

feature_description_ohe: dict = {
    'start_hour': tf.io.VarLenFeature(tf.float32),
    'start_month': tf.io.VarLenFeature(tf.float32),
    'start_date': tf.io.VarLenFeature(tf.float32),
    'day_of_week': tf.io.VarLenFeature(tf.float32)
}


def tf_record_to_jsonl(row):

    dense_input = extract_ohe(row)
    dense_input_flt, fare = extract_flt(row)
    dense_input.extend(dense_input_flt)
    jsonl_line = json.dumps({"dense_input": dense_input, "fare": fare})

    return jsonl_line


def extract_flt(row):
    example = tf.io.parse_single_example(row, feature_description_flt)
    processed_example = {key: value.numpy().tolist() for key, value in example.items()}
    fare = processed_example.pop('fare')

    return list(processed_example.values()), fare


def extract_ohe(row):
    example = tf.io.parse_single_example(row, feature_description_ohe)
    return list(chain.from_iterable(v.values.numpy().tolist() for v in example.values()))


def row_to_tf_example(event):
    if isinstance(event, Row):
        event = event._asdict()

    # Convert the dictionary to a tf.train.Example instance.
    features = {}
    for key, value in event.items():
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.integer):
                features[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value.tolist()))
            elif np.issubdtype(value.dtype, np.floating):
                features[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value.tolist()))
            else:
                raise ValueError(f"Unsupported numpy data type: {value.dtype}")
        elif isinstance(value, str):
            features[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode()]))
        elif isinstance(value, float):
            features[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value]))
        else:
            raise ValueError(f"Unsupported data type: {type(value)} for {key=}")

    return tf.train.Example(
        features=tf.train.Features(feature=features)).SerializeToString()


def get_batch_transform_fn(
        transform_steps: list[TFTOperation],
        artifact_location_dir: str
) -> MLTransform:
    transform_function = MLTransform(write_artifact_location=artifact_location_dir)
    for step in transform_steps:
        transform_function = transform_function.with_transform(step)

    return transform_function


def get_inference_transform_fn(
        artifact_location_dir: str
) -> MLTransform:
    return MLTransform(read_artifact_location=artifact_location_dir)


def add_date_info_fn(element: dict[str, Any]) -> dict[str, Any]:
    # Check if 'trip_start_timestamp' is in the element
    if 'trip_start_timestamp' in element:
        # for json messages timestamps come as string opposed to timestamps from bq
        if isinstance(element['trip_start_timestamp'], str):
            timestamp_format = '%Y-%m-%d %H:%M:%S.%f'
            # Extract the datetime string without the ' UTC' at the end
            datetime_str = element['trip_start_timestamp'].rsplit(' ', 1)[0]
            # Parse the timestamp string to a datetime object
            parsed_timestamp = datetime.strptime(datetime_str, timestamp_format)
            # Since the timestamp is in UTC, attach the UTC timezone to make it timezone-aware
            element['trip_start_timestamp'] = parsed_timestamp.replace(tzinfo=pytz.utc)

        # Extract the day of the week, month, and date
        timestamp = element['trip_start_timestamp']
        day_of_week = timestamp.weekday()
        month = timestamp.month
        date = timestamp.day
        hour = timestamp.hour

        # Add the extracted information to the element
        element['day_of_week'] = str(day_of_week)
        element['start_month'] = str(month)
        element['start_date'] = str(date)
        element['start_hour'] = str(hour)

        # Remove the original 'trip_start_timestamp' field
        del element['trip_start_timestamp']

    return element


def get_prediction(
        instances, endpoint_id):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """

    instances = instances.as_dict()

    # Specify the order of features
    feature_order = ['start_month', 'start_date', 'day_of_week', 'start_hour', 'trip_miles', 'trip_seconds',
                     'dropoff_longitude', 'dropoff_latitude', 'pickup_longitude', 'pickup_latitude']

    # Create the concatenated array based on the specified order
    concatenated_array = np.concatenate([instances[feature].flatten() for feature in feature_order])

    # Ensure the concatenated array is 2D with shape (1, N)
    final_array = concatenated_array.reshape(1, -1)

    # Convert the NumPy array to a list
    dense_input_list = final_array.tolist()[0]

    input_dict_list = [{"dense_input": dense_input_list}]

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
    response = endpoint.predict(instances=input_dict_list)

    # The predictions are a google.protobuf.Value representation of the model's predictions.
    instances['prediction'] = response.predictions

    # convert arrays to list
    combined_data_converted = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in instances.items()}

    # Convert the combined structure to a JSON string
    combined_json = json.dumps(combined_data_converted)

    # Encode this JSON string to bytes
    combined_bytes = combined_json.encode('utf-8')

    return combined_bytes
