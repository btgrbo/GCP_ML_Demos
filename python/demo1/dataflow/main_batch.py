"""
applies transformation to bigquery data and stores result as TFRecord.
Transformation pipeline parameters are stored in GCS bucket.
"""

import argparse
from datetime import datetime

import apache_beam as beam
from apache_beam.io import WriteToTFRecord
from apache_beam.io.gcp.bigquery import ReadFromBigQuery
from apache_beam.ml.transforms import tft
from apache_beam.options.pipeline_options import PipelineOptions

from src import utils
from src.impute_fn import ImputeMean
from src.one_hot_fn import OneHot

# list all transform steps here:
TRANSFORM_STEPS = [
    ImputeMean(columns=['trip_seconds']),
    tft.ScaleTo01(columns=['trip_seconds']),
    OneHot(columns=['payment_type'])
]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bq_table', required=True)
    parser.add_argument('--project', default="bt-int-ml-specialization")
    parser.add_argument('--gcs_bucket',
                        default="gs://bt-int-ml-specialization_dataflow_demo1")
    parser.add_argument('--output_location',
                        default="gs://bt-int-ml-specialization_dataflow_demo1/TFRecords")

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args)

    run_id = f"run_{datetime.utcnow().isoformat()}"

    artifact_location = f"{known_args.gcs_bucket}/transform_artifacts/{run_id}"

    transform_fn = utils.get_batch_transform_fn(TRANSFORM_STEPS, artifact_location)

    read_bq = ReadFromBigQuery(
        table=known_args.bq_table,
        project=known_args.project,
        gcs_location=f"{known_args.gcs_bucket}/bigquery"
    )

    write_tf_record = WriteToTFRecord(
        file_path_prefix=f"{known_args.output_location}/{run_id}",
        file_name_suffix='.tfrecord'
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
                pipeline
                | "ReadFromBigQuery" >> read_bq
                | "Transform" >> transform_fn
                | "ConvertToTFExample" >> beam.Map(utils.row_to_tf_example)
                | "WriteToTFRecord" >> write_tf_record
        )


if __name__ == "__main__":
    main()
