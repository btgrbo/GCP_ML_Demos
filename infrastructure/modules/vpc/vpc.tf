# The actual VPC, a network in which you can create subnetworks for your instances. No subnetworks and routes are created by default
resource "google_compute_network" "vpc" {
  project                         = var.project.name
  name                            = "default"
  auto_create_subnetworks         = false
  delete_default_routes_on_create = true
  description                     = "if no network is specified, most components look for 'default'"
  routing_mode                    = "REGIONAL"
}

resource "google_compute_subnetwork" "vpc" {
  ip_cidr_range            = "172.18.0.0/16"
  name                     = "${google_compute_network.vpc.name}-${var.location}"
  network                  = google_compute_network.vpc.self_link
  region                   = var.location
  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 1
    metadata             = "INCLUDE_ALL_METADATA"
  }
}