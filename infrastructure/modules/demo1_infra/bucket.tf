resource "google_storage_bucket" "dataflow_bucket" {
  name                        = "${var.project.name}_dataflow_demo1"
  location                    = var.location
  uniform_bucket_level_access = true
  labels                      = var.labels

  lifecycle_rule {
    condition {
      age = "60" # Delete runs after 60 days
    }
    action {
      type = "Delete"
    }
  }

}
