resource "google_artifact_registry_repository" "ml" {
  provider = google-beta

  location      = var.location
  repository_id = "ml-${var.name}"
  format        = "DOCKER"
  labels        = var.labels
}
