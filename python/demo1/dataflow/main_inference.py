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

    # TODO: remove all *s after official sdk release
    # using a sdk pre-release where https://github.com/apache/beam/issues/30062 is fixed
    sdk_container_image = "europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/beam_python3.10_sdk:2.54.0rc1"

    pipeline_options = PipelineOptions(
        pipeline_args,
        project=known_args.project_id,            # needs to be set explicitly...
        streaming=True,                           # needed for Pub/Sub
        sdk_location="container",                 # *
        sdk_container_image=sdk_container_image,  # *
    )

    transform_fn = utils.get_inference_transform_fn(
        known_args.transform_artifact_location
    )

    read_pubsub = ReadFromPubSub(subscription=known_args.pubsub_source_subscription)
    write_pubsub = WriteToPubSub(topic=known_args.pubsub_sink_topic)

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
                pipeline
                | "ReadFromPubSub" >> read_pubsub
                | "DecodeJSON" >> beam.Map(lambda x: json.loads(x.decode()))
                | "WindowInto" >> beam.WindowInto(window.FixedWindows(1))  # TODO: why?
                | "AddDateInfo" >> beam.Map(utils.add_date_info_fn)
                | "Transform" >> transform_fn
                | "ConvertToTFExample" >> beam.Map(utils.row_to_tf_example)
                | "SendToModelEndpoint" >> beam.Map(utils.get_prediction)
                | "WriteToPubSub" >> write_pubsub
        )


if __name__ == "__main__":
    main()
