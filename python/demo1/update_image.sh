#!/bin/bash

# To make this script work, you need to install docker and gcloud
# also, you need to login to gcloud using `gcloud auth login` and `gcloud auth application-default login`
# `gcloud auth configure-docker europe-west3-docker.pkg.dev` must be run once to configure docker to use gcloud as a credential helper

set -e  # Exit immediately if a command exits with a non-zero status.
set -x  # Print commands and their arguments as they are executed.
set -u  # Treat unset variables as an error when substituting.
set -o pipefail  # Fail a pipe if any sub-command fails.

IMAGE="europe-west3-docker.pkg.dev/bt-int-ml-specialization/ml-demo1/train:latest"

# change to script's directory
cd "$(dirname "$0")"

# Build the docker image
docker build -t "$IMAGE" .

# Push the docker image to GCR
docker push "$IMAGE"