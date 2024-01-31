variable "project" {
  type = object({
    number = number
    name   = string
  })
}

variable "location" {
  type = string
}