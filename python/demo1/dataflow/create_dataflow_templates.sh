#!/bin/bash
set -o errexit -o nounset -o pipefail

PROJECT="bt-int-ml-specialization"
REGION="europe-west3"

BUCKET_NAME="gs://${PROJECT}_dataflow_demo1"

TEMPLATE_PATH_BATCH="$BUCKET_NAME/templates/demo1-batch.json"
TEMPLATE_IMAGE_BATCH="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_batch:latest"

TEMPLATE_PATH_EVAL="$BUCKET_NAME/templates/demo1-inference.json"
TEMPLATE_IMAGE_EVAL="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_inference:latest"

TEMPLATE_PATH_INFERENCE="$BUCKET_NAME/templates/demo1-eval.json"
TEMPLATE_IMAGE_INFERENCE="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_eval:latest"

gcloud dataflow flex-template build $TEMPLATE_PATH_BATCH \
    --image="$TEMPLATE_IMAGE_BATCH" \
    --sdk-language="PYTHON"

gcloud dataflow flex-template build $TEMPLATE_PATH_EVAL \
    --image="$TEMPLATE_IMAGE_EVAL" \
    --sdk-language="PYTHON"

gcloud dataflow flex-template build $TEMPLATE_PATH_INFERENCE \
    --image="$TEMPLATE_IMAGE_INFERENCE" \
    --sdk-language="PYTHON"
