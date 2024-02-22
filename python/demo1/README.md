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

In order to work with this project the following software is required:
- git
- terraform
- Docker
- Python

## Usage

Before interacting with the GCP from your local environment make sure that you are authenticated:
```bash
gcloud auth login
gloud auth application-default login
```

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
python -m pipeline.py
```

### Changing the Model

In order to alter the model, implement changes in the python/demo1/main.py. Committing these changes to the main branch
of the repo will trigger a cloudbuild run which rebuilds the image and pushes it to the GCP container registry.

### Changing the Preprocessing Pipeline

In order to alter the preprocessing pipeline for either batch or inference, implement these changes in main_batch.py or
main_inference.py. Build the new docker images and push them to the artifact registry

```bash
docker build -f Dockerfile_inf -t europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/dataflow_batch:latest .
docker push europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/dataflow_batch:latest
```
or
```bash
docker build -f Dockerfile_inf -t europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/dataflow_inference:latest .
docker push europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/dataflow_inference:latest
```

To test the batch preprocessing pipeline following changes, run:
```bash
./run_dataflow_batch.sh "YOUR_DATAFLOW_RUN_ID"
```
This will preprocess data for training from the corresponding bigquery table and export the tfiles as tf.records to the
cloud bucket. Additionally, a transform artifact will be saved.

To test the inference preprocessing pipeline following changes, run:
```bash
 ./run_dataflow_inf.sh "PATH_TO_TRANSFORM_ARTIFACT_LOCATION"
```
Once the dataflow job is running you can publish a message to pubsub by running:
```bash
 ./test_dataflow.sh
```
