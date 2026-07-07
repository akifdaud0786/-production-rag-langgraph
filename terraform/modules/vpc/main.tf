variable "project_id" { type = string }
variable "region" { type = string }

resource "google_compute_network" "rag_vpc" {
  name                    = "rag-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id
}

resource "google_compute_subnetwork" "rag_subnet" {
  name          = "rag-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.rag_vpc.id
  project       = var.project_id
}

# Serverless VPC Access connector so Cloud Run services can reach private
# resources (Firestore private endpoints, internal Qdrant, etc.).
resource "google_vpc_access_connector" "connector" {
  name          = "rag-vpc-connector"
  region        = var.region
  network       = google_compute_network.rag_vpc.name
  ip_cidr_range = "10.20.0.0/28"
  project       = var.project_id

  min_instances = 2
  max_instances = 3
}

resource "google_compute_firewall" "allow_internal" {
  name    = "rag-allow-internal"
  network = google_compute_network.rag_vpc.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  source_ranges = ["10.10.0.0/24", "10.20.0.0/28"]
}

output "connector_id" {
  value = google_vpc_access_connector.connector.id
}

output "network_id" {
  value = google_compute_network.rag_vpc.id
}
