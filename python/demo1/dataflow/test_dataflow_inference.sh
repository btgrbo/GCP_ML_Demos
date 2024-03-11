# start a new dataflow run with run_dataflow.sh first.
# Do not forget to drain run after execution...

PROJECT="bt-int-ml-specialization"
PUBSUB_SOURCE_TOPIC="projects/${PROJECT}/topics/demo1-event-source"

MSG='{"trip_start_timestamp": "2015-12-06 17:15:00.000000 UTC",
      "fare": 1,
      "trip_seconds": 540,
      "trip_miles": 2.3,
      "pickup_latitude": 42.009018227,
      "pickup_longitude": -87.672723959,
      "dropoff_latitude": 41.972437081,
      "dropoff_longitude": -87.671109526}'

gcloud pubsub topics publish "$PUBSUB_SOURCE_TOPIC" --message="$MSG"