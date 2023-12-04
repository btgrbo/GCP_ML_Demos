variable "project" {
  type = object({
    number = number
    name   = string
  })
}

variable "labels" {
  type = map(string)
}

variable "demo1_data_readers" {
  type = list(object({
    email = string
  }))
}

variable "demo2_data_readers" {
  type = list(object({
    email = string
  }))
}

variable "data_owners" {
  type = list(string)
}