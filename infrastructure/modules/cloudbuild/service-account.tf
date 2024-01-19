#resource "google_service_account" "cloudbuild" {
#  account_id = "cloudbuild-runner"
#}

#data "google_service_account" "cloudbuild" {
#  project    = var.project.name
#  account_id = "${var.project.number}@cloudbuild.gserviceaccount.com"
#}

data "google_project_service" "cloudbuild" {
  service = "cloudbuild.googleapis.com"
}

resource "google_project_service_identity" "cloudbuild" {
  provider = google-beta

  service = data.google_project_service.cloudbuild.service
}