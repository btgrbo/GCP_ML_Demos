resource "google_storage_bucket" "cloudbuild_bucket" {
  name                        = "${var.project.name}_cloudbuild"
  location                    = var.location
  uniform_bucket_level_access = true
  labels                      = var.labels

  lifecycle_rule {
    condition {
      age = "60" # Delete build source code after 60 days
    }
    action {
      type = "Delete"
    }
  }

}
