"""
applies same transformation that is defined in `main_batch.py` to inference data.
Data source and sink is PubSub, here.
"""

import argparse
import json

import apache_beam as beam
from apache_beam.io.gcp.pubsub import ReadFromPubSub, WriteToPubSub
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import window

from src import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubsub_source_subscription', required=True)
    parser.add_argument('--pubsub_sink_topic', required=True)
    parser.add_argument('--transform_artifact_location', required=True)
    parser.add_argument('--project_id', default="bt-int-ml-specialization")

    known_args, pipeline_args = parser.parse_known_args()

    pipeline_options = PipelineOptions(
        pipeline_args,
        project=known_args.project_id,  # needs to be set explicitly...
        streaming=True,                 # needed for Pub/Sub
    )

    local_artifact_path = f"/tmp/artifacts"  # works on linux only...

    # TODO: remove after https://github.com/apache/beam/issues/30062 has been solved
    #       with new version rollout.
    utils.download_transform_artifacts(
        gcs_path=known_args.transform_artifact_location,
        local_path=local_artifact_path,
        project_id=known_args.project_id,
    )

    transform_fn = utils.get_inference_transform_fn(local_artifact_path)

    read_pubsub = ReadFromPubSub(subscription=known_args.pubsub_source_subscription)
    write_pubsub = WriteToPubSub(topic=known_args.pubsub_sink_topic)

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
                pipeline
                | "ReadFromPubSub" >> read_pubsub
                | "DecodeJSON" >> beam.Map(lambda x: json.loads(x.decode()))
                | "WindowInto" >> beam.WindowInto(window.FixedWindows(1))  # TODO: why?
                | "Transform" >> transform_fn
                | "ConvertToTFExample" >> beam.Map(utils.row_to_tf_example)
                | "WriteToPubSub" >> write_pubsub
        )


if __name__ == "__main__":
    main()
