terraform {
  required_version = ">= 1.3.0, < 2.0.0"
  backend "gcs" {
    bucket = "bt-int-ml-specialization-terraform"
    prefix = "state"
  }
}