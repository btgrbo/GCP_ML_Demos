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