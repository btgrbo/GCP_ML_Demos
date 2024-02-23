output "vertex_executor_sa" {
  value = google_service_account.vertex_executor
}

output "vertex_predictor_sa" {
  value = google_service_account.vertex_predictor
}
