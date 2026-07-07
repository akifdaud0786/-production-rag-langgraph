variable "project_id" { type = string }
variable "region" { type = string }

resource "google_storage_bucket" "raw" {
  name                        = "${var.project_id}-rag-raw"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_storage_bucket" "processed" {
  name                        = "${var.project_id}-rag-processed"
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }
}

output "raw_bucket_name" {
  value = google_storage_bucket.raw.name
}

output "processed_bucket_name" {
  value = google_storage_bucket.processed.name
}
