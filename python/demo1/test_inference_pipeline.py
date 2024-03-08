import textwrap

from google.api_core.exceptions import DeadlineExceeded
from google.cloud import pubsub

PROJECT_ID = "bt-int-ml-specialization"

source_topic = "projects/bt-int-ml-specialization/topics/demo1-event-source"
sink_subscription = "projects/bt-int-ml-specialization/subscriptions/demo1-event-sink-subscription"

MESSAGE = textwrap.dedent(
    """{
        "trip_start_timestamp": "2015-12-06 17:15:00.000000 UTC",
        "fare": 1,
        "trip_seconds": 540,
        "trip_miles": 2.3,
        "pickup_latitude": 42.009018227,
        "pickup_longitude": -87.672723959,
        "dropoff_latitude": 41.972437081,
        "dropoff_longitude": -87.671109526
      }"""
)

if __name__ == "__main__":

    publisher = pubsub.PublisherClient()
    future = publisher.publish(source_topic, data=MESSAGE.encode())
    future.result()

    subscriber = pubsub.SubscriberClient()

    consumed_messages = 0
    try:
        pull_response = subscriber.pull(subscription=sink_subscription, max_messages=10, timeout=5)
        for message in pull_response.received_messages:
            print(f"Received message: {message.message.data.decode('utf-8')}")
            subscriber.acknowledge(subscription=sink_subscription, ack_ids=[message.ack_id])
            consumed_messages += 1
    except DeadlineExceeded:
        ...

    assert consumed_messages == 1, "Num received messages != 1"
