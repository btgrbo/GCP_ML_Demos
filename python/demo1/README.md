# GCP DEMO 1

This GCP Demo project implements an end-to-end TensorFlow pipeline using the Chicago taxi trips dataset. We come up with
a simple deep neural network that predicts taxi fares from temporal and spatial features. To allow price prediction near
realtime, we leverage GCP services for data preprocessing, hyperparameter tuning, model training, and online serving.
This README captures the most relevant information to get started. For a more detailed overview of the project please
read the whitepaper in the documents folder.

## Project Structure

The following diagram outlines the file structure for GCP Demo 1. Note that terraform files relevant for setting up the
cloud infrastructure are not explicitly listed.
```bash
C:.
|   .gitignore                                            # files to be ignored by git
|   cloudbuild.yaml                                       # configuration file for cloudbuild pipeline
|   README.md                                             # the project's documentation
|
+---bigquery                                              # folder for bigquery sql files
|   +---demo1
|   |       taxi_trips_preprocess.sql                     # data exploration and preprocessing in bigquery
|   |
|   \---demo2
|
+---documents
|   |   Machine Learning - Services Specialization Partner Assessment Checklist _ Y23.pdf # Google requirements
|   |   whitepaper_demo1.docx                             # Whitepaper for demo 1
|   |
|   \---figures                                           # exported figures for whitepaper
|           
+---infrastructure                                        # folder for terraform code cloud infrastructure
|
\---python
    +---demo1
    |   |   Dockerfile                                    # Dockerfile for model training
    |   |   local_dev.py                                  # file for local model development
    |   |   local_main_dev.py                             # file for developping main.py locally
    |   |   main.py                                       # model training
    |   |   README.md                                     # README file for demo1
    |   |   requirements.in                               # requirements file to generate requirements.txt 
    |   |   requirements.txt                              # requirements file for setting up the environment
    |   |   update_image.sh                               # file to manually update docker image and push to artifact
    |   |                                                 # registry 
    |   |
    |   +---dataflow                                      # folder for dataflow code
    |   |   |   Dockerfile.runtime                        # creates Dockerimage for inference beam pipeline
    |   |   |   Dockerfile_batch                          # Dockerfile for batch preprocessing
    |   |   |   Dockerfile_inf                            # Dockerfile for inference (online prediction) preprocessing
    |   |   |   main_batch.py                             # main file for batch preprocessing
    |   |   |   main_inference.py                         # main file for inference (online prediction) preprocessing
    |   |   |   requirements.in                           # requirements file to generate requirements.txt 
    |   |   |   requirements.txt                          # requirements file for setting up the environment
    |   |   |   run_dataflow_batch.sh                     # bash script for manually triggering batch preprocessing run 
    |   |   |   run_dataflow_inf.sh                       # bash script for triggering dataflow job for inference
    |   |   |   setup.py                                  # file to make 'src' available in Dockercontainer
    |   |   |   test_dataflow.sh                          # test inference by manually sending message to pubsub
    |   |   |
    |   |   \---src
    |   |       |   one_hot_fn.py                         # one-hot-encoding function for preprocessing
    |   |       |   utils.py                              # preprocessing functions used in beam pipeline
    |   |       |   __init__.py
    |   |
    |   \---pipeline                                      # folder for kubeflow pipeline      
    |       |   pipeline.py                               # kubeflow pipeline triggering a vertex pipeline run for model
    |                                                     # training
    |
    \---demo2
```

## Prerequisites

### Dependency management
In order to add new dependencies `pip-compile` is needed.

### Model Training and Deployment
To start a model training run and deploy the model for online serving trigger the kubeflow pipeline in 
python/demo1/pipeline:
1. Ensure that the environment is installed and activated:
```bash
pip install pipenv
pipenv --python 3.10
pipenv install -r requirements.txt
pipenv shell
```
2. Start the kubeflow pipeline:
```bash
python pipeline/pipeline.py
```

### Deployments

Committing changes to the main branch of the repo will trigger a cloudbuild run that rebuilds all docker images and pushes them to the GCP artifact registry.

The training image can also be uploaded manually. This is only for debugging purposes. To do so run `bash update_image.sh`.

Dataflow flex templates must be set up once. This step is currently executed manually. To create the templates run

```bash
bash dataflow/create_dataflow_templates.sh
```
