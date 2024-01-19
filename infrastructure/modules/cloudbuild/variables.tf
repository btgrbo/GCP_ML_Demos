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