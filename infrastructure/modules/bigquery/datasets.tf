resource "google_bigquery_dataset" "demo1" {
  dataset_id = "demo1"
  location   = "EU"
  project    = var.project.name
  labels     = var.labels
}

resource "google_bigquery_dataset" "demo2" {
  dataset_id = "demo2"
  location   = "EU"
  project    = var.project.name
  labels     = var.labels
}