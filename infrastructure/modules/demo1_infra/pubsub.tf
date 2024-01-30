resource "google_pubsub_topic" "event_source" {
  project = var.project.name
  name    = "demo1-event-source"
}

resource "google_pubsub_subscription" "event_source" {
  name  = "demo1-event-source-subscription"
  topic = google_pubsub_topic.event_source.name

  ack_deadline_seconds       = 20
  message_retention_duration = "1200s"

}

#############################################################


resource "google_pubsub_topic" "event_sink" {
  project = var.project.name
  name    = "demo1-event-sink"
}

resource "google_pubsub_subscription" "event_sink" {
  name  = "demo1-event-sink"
  topic = google_pubsub_topic.event_sink.name

  ack_deadline_seconds       = 20
  message_retention_duration = "1200s"

}