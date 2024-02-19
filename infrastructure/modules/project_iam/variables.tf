variable "project" {
  type = object({
    number = number
    name   = string
  })
}

variable "project_admins" {
  type = list(string)
}

variable "aiplatform_service" {
  type = object({
    service = string
  })
}

variable "vertex_executors" {
  type = list(object({
    email = string
  }))
}

variable "cloudbuild_sa" {
  type = object({
    email = string
  })
}

variable "dataflow_accounts" {
  type = list(object({
    email = string
  }))
}

variable "logger_accounts" {
  type = list(object({
    email = string
  }))
}
