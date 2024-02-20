#############################################################
# pubsub event_source

resource "google_pubsub_topic_iam_binding" "event_source_topic" {
  members = [for sa in var.publishers : "serviceAccount:${sa.email}"]
  role    = "roles/pubsub.publisher"
  topic   = google_pubsub_topic.event_source.name
}

resource "google_pubsub_subscription_iam_binding" "event_source_subscription" {
  members      = ["serviceAccount:${google_service_account.dataflow_inference.email}"]
  role         = "roles/pubsub.subscriber"
  subscription = google_pubsub_subscription.event_source.name
}

#############################################################
# pubsub event_sink

resource "google_pubsub_topic_iam_binding" "event_sink_topic" {
  members = ["serviceAccount:${google_service_account.dataflow_inference.email}"]
  role    = "roles/pubsub.publisher"
  topic   = google_pubsub_topic.event_sink.name
}

resource "google_pubsub_subscription_iam_binding" "event_sink_subscription" {
  members      = [for sa in var.subscribers : "serviceAccount:${sa.email}"]
  role         = "roles/pubsub.subscriber"
  subscription = google_pubsub_subscription.event_sink.name
}

#############################################################
# service account invoker

data "google_iam_policy" "service_account" {
  binding {
    role    = "roles/iam.serviceAccountTokenCreator"
    members = [for sa in var.dataflow_invokers : "serviceAccount:${sa.email}"]
  }
  binding {
    members = [for sa in var.dataflow_invokers : "serviceAccount:${sa.email}"]
    role    = "roles/iam.serviceAccountUser"
  }
}

resource "google_service_account_iam_policy" "dataflow_inference_service_account" {
  service_account_id = google_service_account.dataflow_inference.id
  policy_data        = data.google_iam_policy.service_account.policy_data
}

resource "google_service_account_iam_policy" "dataflow_batch_service_account" {
  service_account_id = google_service_account.dataflow_batch.id
  policy_data        = data.google_iam_policy.service_account.policy_data
}

#############################################################
# dataflow bucket

data "google_iam_policy" "storage" {
  binding {
    role    = "roles/storage.objectAdmin"
    members = concat(
      [
        "serviceAccount:${google_service_account.dataflow_batch.email}",
        "serviceAccount:${google_service_account.dataflow_inference.email}",
        "serviceAccount:service-${var.project.number}@dataflow-service-producer-prod.iam.gserviceaccount.com",
      ],
      [for sa in var.dataflow_invokers : "serviceAccount:${sa.email}"],
      var.admins,
    )
  }
}


resource "google_storage_bucket_iam_policy" "storage" {
  bucket      = google_storage_bucket.dataflow_bucket.name
  policy_data = data.google_iam_policy.storage.policy_data
}
