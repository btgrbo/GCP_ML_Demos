data "google_iam_policy" "storage" {
  binding {
    role    = "roles/storage.objectAdmin"
    members = concat(
      ["serviceAccount:${google_project_service_identity.cloudbuild.email}"],
      var.admins,
    )
  }
}


resource "google_storage_bucket_iam_policy" "storage" {
  bucket      = google_storage_bucket.cloudbuild_bucket.name
  policy_data = data.google_iam_policy.storage.policy_data
}