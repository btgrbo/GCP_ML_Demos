output "dataflow_batch_sa" {
  value = google_service_account.dataflow_batch
}

output "dataflow_inference_sa" {
  value = google_service_account.dataflow_inference
}