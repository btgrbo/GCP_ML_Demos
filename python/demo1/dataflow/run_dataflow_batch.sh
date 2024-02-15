#!/bin/bash
set -o errexit -o nounset -o pipefail

PROJECT="bt-int-ml-specialization"
REGION="europe-west3"

BUCKET_NAME="gs://${PROJECT}_dataflow_demo1"
TEMPLATE_PATH="$BUCKET_NAME/templates/demo1-batch.json"
TEMPLATE_IMAGE="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_batch:latest"

gcloud dataflow flex-template build $TEMPLATE_PATH \
    --image="$TEMPLATE_IMAGE" \
    --sdk-language="PYTHON"

gcloud dataflow flex-template run "demo1-batch-`date +%Y%m%d-%H%M%S`" \
    --disable-public-ips \
    --max-workers=1 \
    --num-workers=1 \
    --parameters project_id="$PROJECT" \
    --project="$PROJECT" \
    --region="$REGION" \
    --service-account-email="d1-dataflow-batch-runner@$PROJECT.iam.gserviceaccount.com" \
    --staging-location="$BUCKET_NAME/batch/staging" \
    --subnetwork="https://www.googleapis.com/compute/v1/projects/$PROJECT/regions/$REGION/subnetworks/default-$REGION" \
    --temp-location="$BUCKET_NAME/batch/temp" \
    --template-file-gcs-location="$TEMPLATE_PATH" \
    --worker-machine-type="n1-standard-2"