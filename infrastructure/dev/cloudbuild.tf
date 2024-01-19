module "cloudbuild" {
  source = "../modules/cloudbuild"

  labels   = local.labels
  location = local.location
  project  = data.google_project.default
  admins   = local.admins

  depends_on = [
    module.services.cloudbuild,
  ]

}
