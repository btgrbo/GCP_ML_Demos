module "vpc" {
  source = "../modules/vpc"

  project  = data.google_project.default
  location = "europe-west3"
}