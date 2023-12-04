data "google_iam_policy" "storage" {
  binding {
    role    = "roles/storage.objectAdmin"
    members = concat(
      ["serviceAccount:${google_service_account.vertex_executor.email}"],
      var.admins,
    )
  }
  binding {
    role    = "roles/storage.objectViewer"
    members = ["serviceAccount:${google_service_account.vertex_predictor.email}"]
  }
}


resource "google_storage_bucket_iam_policy" "storage" {
  bucket      = google_storage_bucket.ml_pipeline_bucket.name
  policy_data = data.google_iam_policy.storage.policy_data
}


data "google_iam_policy" "act_as_predictor" {
  binding {
    role    = "roles/iam.serviceAccountUser"
    members = ["serviceAccount:${google_service_account.vertex_executor.email}"]
  }
}

resource "google_service_account_iam_policy" "admin-account-iam" {
  service_account_id = google_service_account.vertex_predictor.name
  policy_data        = data.google_iam_policy.act_as_predictor.policy_data
}

data "google_iam_policy" "vertex_ai_runner_service_account" {
  binding {
    role = "roles/iam.serviceAccountTokenCreator"
    members = var.admins
  }
}

resource "google_service_account_iam_policy" "vertex_ai_runner_service_account" {
  service_account_id = google_service_account.vertex_executor.id
  policy_data        = data.google_iam_policy.vertex_ai_runner_service_account.policy_data
}