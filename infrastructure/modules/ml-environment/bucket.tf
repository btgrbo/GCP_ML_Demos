resource "google_storage_bucket" "ml_pipeline_bucket" {
  name                        = "${var.project.name}-ml-${var.name}"
  location                    = var.location
  uniform_bucket_level_access = true
  labels                      = var.labels
}
