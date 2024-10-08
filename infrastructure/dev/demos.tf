# demo 1

module "demo1" {
  source = "../modules/ml-environment"

  name             = "demo1"
  labels           = local.labels
  location         = local.location
  project          = data.google_project.default
  admins           = local.admins
  artifact_readers = [module.demo1_infra.dataflow_batch_sa, module.demo1_infra.dataflow_inference_sa]
  artifact_writers = [module.cloudbuild.cloudbuild_sa]

  depends_on = [
    module.services.aiplatform,
    module.services.artifactregistry,
  ]
}


module "demo1_infra" {
  source = "../modules/demo1_infra"

  labels            = local.labels
  location          = local.location
  project           = data.google_project.default
  admins            = local.admins
  publishers        = []
  subscribers       = []
  dataflow_invokers = [module.demo1.vertex_executor_sa]

  depends_on = [
    module.services.aiplatform,
    module.services.artifactregistry,
  ]
}

########################################################################
# demo 2

module "demo2" {
  source = "../modules/ml-environment"

  name             = "demo2"
  labels           = local.labels
  location         = local.location
  project          = data.google_project.default
  admins           = local.admins
  artifact_readers = []
  artifact_writers = [module.cloudbuild.cloudbuild_sa]

  depends_on = [
    module.services.aiplatform,
    module.services.artifactregistry,
  ]
}
