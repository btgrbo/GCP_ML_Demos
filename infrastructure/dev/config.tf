locals {
  labels = {
    "env" = "dev"
  }
  location = "europe-west3"
  admins = [
    "group:bt-int-ml-specialization@btelligent.com",
    "user:laurenz.reitsam@btelligent.com",
    "user:gregory.born@btelligent.com",
    "user:oliver.nowak@btelligent.com",
  ]
}

data "google_project" "default" {
  project_id = "bt-int-ml-specialization"
}