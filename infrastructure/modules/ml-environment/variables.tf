variable "name" {
  type = string
}

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

variable "admins" {
    type = list(string)
}

variable "artifact_writers" {
    type = list(object({email: string}))
}

variable "artifact_readers" {
    type = list(object({email: string}))
}