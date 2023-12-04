resource "google_service_account" "vertex_executor" {
  account_id = "ml-${var.name}-executor"
}

resource "google_service_account" "vertex_predictor" {
  account_id = "ml-${var.name}-predictor"
}