variable "project" {
  type = object({
    number = number
    name   = string
  })
}

variable "location" {
  type = string
}

variable "labels" {
  type = map(string)
}

variable "publishers" {
  type = list(object({
    email = string
  }))
}

variable "subscribers" {
  type = list(object({
    email = string
  }))
}

variable "dataflow_invokers" {
  type = list(object({
    email = string
  }))
}

variable "admins" {
    type = list(string)
}