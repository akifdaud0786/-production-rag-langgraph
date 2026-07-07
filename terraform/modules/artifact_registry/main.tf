variable "project_id" { type = string }
variable "region" { type = string }

resource "google_artifact_registry_repository" "rag_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = "rag-repo"
  description   = "Container images for the production RAG system (API + frontend)"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-last-10"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
}

output "repository_id" {
  value = google_artifact_registry_repository.rag_repo.repository_id
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.rag_repo.repository_id}"
}
