resource "google_project_service" "artifactregistry" {
  service = "artifactregistry.googleapis.com"
}

resource "google_project_service" "aiplatform" {
  service = "aiplatform.googleapis.com"
}

resource "google_project_service" "bigquery" {
  service = "bigquery.googleapis.com"
}