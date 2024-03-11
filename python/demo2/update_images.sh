#!/bin/bash

# To make this script work, you need to install docker and gcloud
# also, you need to login to gcloud using `gcloud auth login` and `gcloud auth application-default login`
# `gcloud auth configure-docker europe-west3-docker.pkg.dev` must be run once to configure docker to use gcloud as a credential helper

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.
set -u  # Treat unset variables as an error when substituting.
set -o pipefail  # Fail a pipe if any sub-command fails.

IMAGE_TRAINING="europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/train:latest"
IMAGE_PREDICTION="europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo2/prediction:latest"



# change to script's directory
cd "$(dirname "$0")"

# Build the docker images
docker build --platform=linux/amd64 -t "$IMAGE_TRAINING" -f ./Dockerfile.training .
docker build --platform=linux/amd64 -t "$IMAGE_PREDICTION" -f ./Dockerfile.prediction .

# Push the docker images to GCR
docker push "$IMAGE_TRAINING"
docker push "$IMAGE_PREDICTION"