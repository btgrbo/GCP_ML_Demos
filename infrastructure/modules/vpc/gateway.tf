# A route that sends traffic to the public internet through the default internet gateway of google
resource "google_compute_route" "internet_gw" {
  dest_range       = "0.0.0.0/0" # public internet
  name             = "${google_compute_network.vpc.name}-internet-gateway"
  network          = google_compute_network.vpc.self_link
  next_hop_gateway = "default-internet-gateway"
}

# A router that is used for NAT for instances with only internal IPs, which enables internet access for those instances
resource "google_compute_router" "internet_gw" {
  name    = "${google_compute_network.vpc.name}-internet-gateway"
  network = google_compute_network.vpc.self_link
}

# The IP that instances with only internal IPs will have in the internet, also used for cross-project internet communication
resource "google_compute_address" "external_nat_ip" {
  name         = "${google_compute_network.vpc.name}-external-nat-ip"
  region       = var.location
  network_tier = "PREMIUM"
}

# The actual NAT-Mechanism connecting the router and the external_nat_ip
resource "google_compute_router_nat" "internet_gw" {
  name                               = "${google_compute_network.vpc.name}-internet-gw"
  region                             = var.location
  router                             = google_compute_router.internet_gw.name
  nat_ip_allocate_option             = "MANUAL_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  nat_ips                            = [google_compute_address.external_nat_ip.self_link]
}