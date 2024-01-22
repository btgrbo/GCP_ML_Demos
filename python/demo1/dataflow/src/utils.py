import json
import re
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from apache_beam import Row
from apache_beam.ml import MLTransform
from apache_beam.ml.transforms.tft import TFTOperation
from google.cloud import storage


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


def _replace_artifact_location(attributes_json: Path):
    """this is needed because stupidity"""
    pth = attributes_json.parent
    content = json.loads(attributes_json.read_text())
    artifact_location = content[0]["artifact_location"]
    content[0]["artifact_location"] = str(pth / artifact_location.split("/")[-1])
    attributes_json.write_text(json.dumps(content))


def download_transform_artifacts(gcs_path: str, local_path: str, project_id: str):
    """using artifacts directly from GCS seems to be broken in READ mode"""

    local_path = Path(local_path)
    shutil.rmtree(local_path, ignore_errors=True)
    local_path.mkdir(parents=True)

    client = storage.Client(project=project_id)

    match = re.match(r'gs://([^/]+)/(.+)', gcs_path)
    if match is None:
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    bucket_name, path = match.groups()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path)

    for blob in blobs:
        local_file_path = local_path / blob.name.replace(path, "")
        if blob.name.endswith('/') and blob.size == 0:  # directory
            local_file_path.mkdir()
        else:
            blob.download_to_filename(local_file_path)  # file

    _replace_artifact_location(local_path / "attributes.json")
