output "api_service_url" {
  value       = module.cloud_run_api.service_url
  description = "Public URL of the FastAPI RAG service"
}

output "frontend_service_url" {
  value       = module.cloud_run_frontend.service_url
  description = "Public URL of the Streamlit chat UI"
}

output "raw_bucket_name" {
  value = module.gcs.raw_bucket_name
}

output "processed_bucket_name" {
  value = module.gcs.processed_bucket_name
}

output "artifact_registry_repo" {
  value = module.artifact_registry.repository_id
}

output "vpc_connector_id" {
  value = module.vpc.connector_id
}
