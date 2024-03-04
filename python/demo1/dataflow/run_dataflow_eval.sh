#!/bin/bash
set -o errexit -o nounset -o pipefail

PROJECT="bt-int-ml-specialization"
REGION="europe-west3"

BUCKET_NAME="gs://${PROJECT}_dataflow_demo1"
TEMPLATE_PATH="$BUCKET_NAME/templates/demo1-eval.json"
TEMPLATE_IMAGE="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_eval:latest"

gcloud dataflow flex-template build $TEMPLATE_PATH \
    --image="$TEMPLATE_IMAGE" \
    --sdk-language="PYTHON"