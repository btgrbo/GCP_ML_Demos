resource "google_service_account" "dataflow_inference" {
  account_id = "d1-dataflow-inference-runner"
}

resource "google_service_account" "dataflow_batch" {
  account_id = "d1-dataflow-batch-runner"
}