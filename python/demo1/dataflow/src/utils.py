from typing import Any

import numpy as np
import tensorflow as tf
from apache_beam import Row
from apache_beam.ml import MLTransform
from apache_beam.ml.transforms.tft import TFTOperation


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