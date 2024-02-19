module "proejct_iam" {
  source = "../modules/project_iam"

  project            = data.google_project.default
  project_admins     = local.admins
  aiplatform_service = module.services.aiplatform
  vertex_executors   = [module.demo1.vertex_executor_sa, module.demo2.vertex_executor_sa]
  cloudbuild_sa      = module.cloudbuild.cloudbuild_sa
  dataflow_accounts  = [module.demo1_infra.dataflow_batch_sa, module.demo1_infra.dataflow_inference_sa]
  logger_accounts    = [module.demo1.vertex_predictor_sa, module.demo2.vertex_predictor_sa]
}
