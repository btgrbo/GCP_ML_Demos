# start a new dataflow run with run_dataflow.sh first.
# Do not forget to drain run after execution...

PROJECT="bt-int-ml-specialization"
PUBSUB_SOURCE_TOPIC="projects/${PROJECT}/topics/demo1-event-source"

MSG='{"unique_key": "6d0b", "trip_seconds": 123.0, "payment_type": "Cash"}'

gcloud pubsub topics publish "$PUBSUB_SOURCE_TOPIC" --message="$MSG"