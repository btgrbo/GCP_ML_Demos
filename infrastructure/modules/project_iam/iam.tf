resource "google_project_service_identity" "aiplatform" {
  provider = google-beta
  service  = var.aiplatform_service.service
}

data "google_iam_policy" "project_iam" {
  binding {
    role    = "roles/storage.admin"
    members = var.project_admins
  }
  binding {
    role    = "roles/storage.objectAdmin"
    members = var.project_admins
  }
  binding {
    role    = "roles/owner"
    members = var.project_admins
  }
#  binding {
#    role    = "roles/viewer"
#    members = [for sa in var.dataflow_accounts : "serviceAccount:${sa.email}"]
#  }
  binding {
    role    = "roles/bigquery.dataOwner"
    members = var.project_admins
  }
  binding {
    role    = "roles/aiplatform.serviceAgent"
    members = [
      "serviceAccount:${google_project_service_identity.aiplatform.email}",
    ]
  }
  binding {
    role    = "roles/aiplatform.admin"
    members = ["serviceAccount:service-${var.project.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"]
  }
  binding {
    role    = "roles/aiplatform.user"
    members = concat([for sa in var.vertex_executors : "serviceAccount:${sa.email}"],
                     [for sa in var.dataflow_accounts : "serviceAccount:${sa.email}"]
    )
  }
  binding {
    role    = "roles/aiplatform.customCodeServiceAgent"
    members = [
      "serviceAccount:service-${var.project.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com",
      # "serviceAccount:service-${google_project_service_identity.aiplatform.email}",
    ]
  }
  binding {
    role    = "roles/logging.logWriter"
    members = concat(
      [for sa in var.vertex_executors : "serviceAccount:${sa.email}"],
      [for sa in var.dataflow_accounts : "serviceAccount:${sa.email}"],
      ["serviceAccount:${var.cloudbuild_sa.email}"],
    )
  }
  binding {
    role    = "roles/cloudbuild.builds.builder"
    members = ["serviceAccount:${var.cloudbuild_sa.email}"]
  }
  binding {
    role    = "roles/cloudbuild.workerPoolUser"
    members = ["serviceAccount:${var.cloudbuild_sa.email}"]
  }
  binding {
    role    = "roles/cloudbuild.serviceAgent"
    members = ["serviceAccount:${var.cloudbuild_sa.email}"]
  }
  binding {
    role    = "roles/cloudbuild.serviceAgent"
    members = ["serviceAccount:service-${var.project.number}@gcp-sa-cloudbuild.iam.gserviceaccount.com"]
  }
  binding {
    role    = "roles/dataflow.serviceAgent"
    members = ["serviceAccount:service-${var.project.number}@dataflow-service-producer-prod.iam.gserviceaccount.com"]
  }
  binding {
    role    = "roles/dataflow.worker"
    members = [for sa in var.dataflow_accounts : "serviceAccount:${sa.email}"]
  }
  binding {
    role    = "roles/dataflow.admin"
    members = concat(
      [for sa in var.vertex_executors : "serviceAccount:${sa.email}"]
    )
  }
  binding {
    role    = "roles/bigquery.jobUser"
    members = ["serviceAccount:${var.dataflow_accounts[0].email}"]
  }
}

resource "google_project_iam_policy" "admins" {
  project     = var.project.name
  policy_data = data.google_iam_policy.project_iam.policy_data
}
