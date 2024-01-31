#!/bin/bash
set -o errexit -o nounset -o pipefail

TRANSFORM_ARTIFACT_LOCATION=$1

PROJECT="bt-int-ml-specialization"
REGION="europe-west3"

PUBSUB_SOURCE_SUBSCRIPTION="projects/${PROJECT}/subscriptions/demo1-event-source-subscription"
PUBSUB_SINK_TOPIC="projects/${PROJECT}/topics/demo1-event-sink"

BUCKET_NAME="gs://${PROJECT}_dataflow_demo1"
TEMPLATE_PATH="$BUCKET_NAME/templates/demo1-inference.json"
TEMPLATE_IMAGE="$REGION-docker.pkg.dev/$PROJECT/ml-demo1/dataflow_inference:latest"

gcloud dataflow flex-template build $TEMPLATE_PATH \
    --image "$TEMPLATE_IMAGE" \
    --sdk-language "PYTHON"

gcloud dataflow flex-template run "demo1-inference-`date +%Y%m%d-%H%M%S`" \
    --disable-public-ips \
    --max-workers 1 \
    --num-workers 1 \
    --parameters project_id="$PROJECT" \
    --parameters pubsub_sink_topic="$PUBSUB_SINK_TOPIC" \
    --parameters pubsub_source_subscription="$PUBSUB_SOURCE_SUBSCRIPTION" \
    --parameters transform_artifact_location="$TRANSFORM_ARTIFACT_LOCATION" \
    --project "$PROJECT" \
    --region "$REGION" \
    --service-account-email "d1-dataflow-inference-runner@$PROJECT.iam.gserviceaccount.com" \
    --staging-location "$BUCKET_NAME/inference/staging" \
    --subnetwork "https://www.googleapis.com/compute/v1/projects/$PROJECT/regions/$REGION/subnetworks/default-$REGION" \
    --temp-location "$BUCKET_NAME/inference/temp" \
    --template-file-gcs-location "$TEMPLATE_PATH" \
    --worker-machine-type="n1-standard-2"