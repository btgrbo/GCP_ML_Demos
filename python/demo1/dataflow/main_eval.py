"""
applies transformation to bigquery data and stores result as JSONL.
Transformation pipeline parameters are stored in GCS bucket.
"""

import argparse
import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.io.gcp.bigquery import ReadFromBigQuery
from apache_beam.options.pipeline_options import PipelineOptions
from src import utils

import time

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_run', required=True)
    parser.add_argument('--bq_table', default='bt-int-ml-specialization.demo1.taxi_trips_eval')
    parser.add_argument('--project_id', default="bt-int-ml-specialization")
    parser.add_argument('--gcs_bucket', default="gs://bt-int-ml-specialization_dataflow_demo1")
    parser.add_argument('--output_location',
                        default="gs://bt-int-ml-specialization_dataflow_demo1/jsonl_files")
    parser.add_argument('--transform_artifact_location', required=True)

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(pipeline_args,
                                       project=known_args.project_id  # needs to be set explicitly...
                                       )

    transform_fn = utils.get_inference_transform_fn(known_args.transform_artifact_location)

    read_bq = ReadFromBigQuery(
        table=known_args.bq_table,
        project=known_args.project_id,
        gcs_location=f"{known_args.gcs_bucket}/bigquery"
    )

    write_jsonl_file = WriteToText(
            file_path_prefix=f"{known_args.output_location}/{known_args.df_run}",
            file_name_suffix='.jsonl',
            shard_name_template='') # Avoid adding shard suffixes for single file

    start = time.time()
    print('starting pipeline', start)

    class Printer(beam.DoFn):
        def process(self, element):
            print(start - time.time(), element)
            yield element

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
                pipeline
                | "ReadFromBigQuery" >> read_bq
                | "Print after ReadFromBigQuery" >> beam.ParDo(Printer())

                | "AddDateInfo" >> beam.Map(utils.add_date_info_fn)
                | "Print after AddDateInfo" >> beam.ParDo(Printer())

                | "Transform" >> transform_fn
                | "Print after Transform" >> beam.ParDo(Printer())

                | "ConvertToTFExample" >> beam.Map(utils.row_to_tf_example)
                | "Print after ConvertToTFExample" >> beam.ParDo(Printer())

                | "ConvertToJSON" >> beam.Map(utils.tf_record_to_jsonl)
                | "Print after ConvertToJSON" >> beam.ParDo(Printer())

                | "WriteToGCS" >> write_jsonl_file
                | "Print after WriteToGCS" >> beam.ParDo(Printer())
        )


if __name__ == "__main__":
    main()
