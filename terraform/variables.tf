variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "groq_api_key" {
  description = "Groq API key, stored in Secret Manager"
  type        = string
  sensitive   = true
}

variable "groq_primary_model" {
  type    = string
  default = "llama-3.3-70b-versatile"
}

variable "groq_fallback_model" {
  type    = string
  default = "llama-3.1-8b-instant"
}

variable "api_image" {
  description = "Fully-qualified container image for the API service (built by cloudbuild.yaml)"
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/rag-repo/rag-api:latest"
}

variable "frontend_image" {
  description = "Fully-qualified container image for the frontend service"
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/rag-repo/rag-frontend:latest"
}

variable "vector_backend" {
  description = "qdrant (external/dev) or vertex (production)"
  type        = string
  default     = "vertex"
}

variable "enable_vertex_vector_search" {
  description = "Whether to provision Vertex AI Vector Search infra (costly; disable for early dev)"
  type        = bool
  default     = false
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "max_instances" {
  type    = number
  default = 5
}
