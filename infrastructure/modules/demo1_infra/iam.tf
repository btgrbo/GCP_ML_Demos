#############################################################
# pubsub event_source

data "google_iam_policy" "event_source" {
  binding {
    role    = "roles/pubsub.publisher"
    members = [for sa in var.publishers : "serviceAccount:${sa.email}"]
  }
  binding {
    role    = "roles/pubsub.subscriber"
    members = ["serviceAccount:${google_service_account.dataflow_inference.email}"]
  }
}

resource "google_pubsub_topic_iam_policy" "event_source" {
  topic       = google_pubsub_topic.event_source.name
  policy_data = data.google_iam_policy.event_source.policy_data
}

#############################################################
# pubsub event_sink

data "google_iam_policy" "event_sink" {
  binding {
    role    = "roles/pubsub.publisher"
    members = ["serviceAccount:${google_service_account.dataflow_inference.email}"]
  }
  binding {
    role    = "roles/pubsub.subscriber"
    members = [for sa in var.subscribers : "serviceAccount:${sa.email}"]
  }
}

resource "google_pubsub_topic_iam_policy" "event_sink" {
  topic       = google_pubsub_topic.event_sink.name
  policy_data = data.google_iam_policy.event_sink.policy_data
}

#############################################################
# service account invoker

data "google_iam_policy" "service_account" {
  binding {
    role    = "roles/iam.serviceAccountTokenCreator"
    members = [for sa in var.dataflow_invokers : "serviceAccount:${sa.email}"]
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
      ],
      var.admins,
    )
  }
}


resource "google_storage_bucket_iam_policy" "storage" {
  bucket      = google_storage_bucket.dataflow_bucket.name
  policy_data = data.google_iam_policy.storage.policy_data
}