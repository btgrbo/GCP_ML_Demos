# GCP ML Demos
- [Demo 1 - New York Taxi Trips](./python/demo1/README.md)
- [Demo 2 - Black Friday](./python/demo1/README.md)
- Demo 3 - XRay AutoML

# Getting Started
1. Software dependencies
    - terraform
    - python 3.10
    - docker

2. Setup

Before interacting with the GCP from your local environment make sure that you are authenticated:
```bash
gcloud auth login
gloud auth application-default login
```

The following step shall be executed onece to authorize docker
```bash
gcloud auth configure-docker europe-west3-docker.pkg.dev
```