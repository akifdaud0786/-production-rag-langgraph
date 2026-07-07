variable "project_id" { type = string }
variable "region" { type = string }
variable "index_name" { type = string }
variable "dimensions" {
  type    = number
  default = 384
}

resource "google_vertex_ai_index" "rag_index" {
  project      = var.project_id
  region       = var.region
  display_name = var.index_name
  description  = "Vector index for RAG document chunks (production backend)"

  metadata {
    contents_delta_uri = "gs://${var.project_id}-rag-processed/vertex_index_data/"
    config {
      dimensions                   = var.dimensions
      approximate_neighbors_count  = 50
      distance_measure_type        = "COSINE_DISTANCE"
      shard_size                   = "SHARD_SIZE_MEDIUM"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  index_update_method = "STREAM_UPDATE"
}

resource "google_vertex_ai_index_endpoint" "rag_index_endpoint" {
  project      = var.project_id
  region       = var.region
  display_name = "${var.index_name}-endpoint"
  description  = "Public endpoint serving the RAG vector index"

  public_endpoint_enabled = true
}

resource "google_vertex_ai_index_endpoint_deployed_index" "deployed" {
  index_endpoint    = google_vertex_ai_index_endpoint.rag_index_endpoint.id
  index             = google_vertex_ai_index.rag_index.id
  deployed_index_id = replace(var.index_name, "-", "_")
  display_name      = "${var.index_name}-deployed"

  dedicated_resources {
    machine_spec {
      machine_type = "e2-standard-2"
    }
    min_replica_count = 1
    max_replica_count = 2
  }
}

output "index_id" {
  value = google_vertex_ai_index.rag_index.id
}

output "index_endpoint_id" {
  value = google_vertex_ai_index_endpoint.rag_index_endpoint.id
}

output "deployed_index_id" {
  value = google_vertex_ai_index_endpoint_deployed_index.deployed.deployed_index_id
}
