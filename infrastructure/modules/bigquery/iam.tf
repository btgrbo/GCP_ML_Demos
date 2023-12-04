data "google_iam_policy" "demo1" {
  binding {
    role = "roles/bigquery.dataViewer"
    members = [for sa in var.demo1_data_readers: "serviceAccount:${sa.email}"]
  }
  binding {
    role = "roles/bigquery.dataOwner"
    members = var.data_owners
  }
}

resource "google_bigquery_dataset_iam_policy" "demo1" {
  dataset_id = google_bigquery_dataset.demo1.dataset_id
  policy_data = data.google_iam_policy.demo1.policy_data
}

# ----------------------------------------------------------------------

data "google_iam_policy" "demo2" {
  binding {
    role = "roles/bigquery.dataViewer"
    members = [for sa in var.demo1_data_readers: "serviceAccount:${sa.email}"]
  }
  binding {
      role = "roles/bigquery.dataOwner"
      members = var.data_owners
  }
}

resource "google_bigquery_dataset_iam_policy" "demo2" {
  dataset_id = google_bigquery_dataset.demo2.dataset_id
  policy_data = data.google_iam_policy.demo2.policy_data
}