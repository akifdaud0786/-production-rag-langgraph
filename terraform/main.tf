terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.10"
    }
  }

  backend "gcs" {
    # Configure via: terraform init -backend-config="bucket=YOUR_TF_STATE_BUCKET"
    prefix = "terraform/state/production-rag-langgraph"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- Enable required APIs ---
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "vpcaccess.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# --- Networking ---
module "vpc" {
  source     = "./modules/vpc"
  project_id = var.project_id
  region     = var.region
  depends_on = [google_project_service.required]
}

# --- Storage buckets (raw + processed docs) ---
module "gcs" {
  source     = "./modules/gcs"
  project_id = var.project_id
  region     = var.region
  depends_on = [google_project_service.required]
}

# --- Artifact Registry for container images ---
module "artifact_registry" {
  source     = "./modules/artifact_registry"
  project_id = var.project_id
  region     = var.region
  depends_on = [google_project_service.required]
}

# --- Secret for the Groq API key ---
resource "google_secret_manager_secret" "groq_api_key" {
  secret_id = "groq-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "groq_api_key_value" {
  secret      = google_secret_manager_secret.groq_api_key.id
  secret_data = var.groq_api_key
}

# --- Cloud Run: API service ---
module "cloud_run_api" {
  source                = "./modules/cloud_run"
  project_id            = var.project_id
  region                = var.region
  service_name          = "rag-api"
  image                 = var.api_image
  vpc_connector_id      = module.vpc.connector_id
  cpu                   = "2"
  memory                = "2Gi"
  min_instances         = var.min_instances
  max_instances         = var.max_instances
  allow_unauthenticated = true
  env_vars = {
    VECTOR_BACKEND            = var.vector_backend
    GCP_PROJECT_ID             = var.project_id
    GCP_REGION                 = var.region
    GCS_RAW_BUCKET             = module.gcs.raw_bucket_name
    GCS_PROCESSED_BUCKET       = module.gcs.processed_bucket_name
    GROQ_PRIMARY_MODEL         = var.groq_primary_model
    GROQ_FALLBACK_MODEL        = var.groq_fallback_model
    GUARDRAILS_ENABLED         = "true"
  }
  secret_env_vars = {
    GROQ_API_KEY = google_secret_manager_secret.groq_api_key.secret_id
  }
  depends_on = [google_project_service.required]
}

# --- Cloud Run: Streamlit frontend ---
module "cloud_run_frontend" {
  source                = "./modules/cloud_run"
  project_id            = var.project_id
  region                = var.region
  service_name          = "rag-frontend"
  image                 = var.frontend_image
  vpc_connector_id      = module.vpc.connector_id
  cpu                   = "1"
  memory                = "1Gi"
  min_instances         = 0
  max_instances         = var.max_instances
  allow_unauthenticated = true
  env_vars = {
    API_URL = module.cloud_run_api.service_url
  }
  secret_env_vars = {}
  depends_on = [module.cloud_run_api]
}

# --- Vertex AI Vector Search (production vector store) ---
# Only provisioned when enable_vertex_vector_search = true (skip for Qdrant-based dev/staging).
module "vertex_vector_search" {
  source     = "./modules/vertex_vector_search"
  count      = var.enable_vertex_vector_search ? 1 : 0
  project_id = var.project_id
  region     = var.region
  index_name = "rag-document-index"
  dimensions = 384
  depends_on = [google_project_service.required]
}
