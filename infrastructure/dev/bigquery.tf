module "bigquery" {
  source = "../modules/bigquery"

  project            = data.google_project.default
  labels             = local.labels
  demo1_data_readers = [module.demo1.vertex_executor_sa]
  demo2_data_readers = [module.demo2.vertex_executor_sa]
  data_owners        = local.admins
}
