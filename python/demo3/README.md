# Demo 3 - Brain MRI for Brain Tumor Detection

This demo implements the machine learning solution to solve the [Brain MRI for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) classification problem.

## Structure

### Pipeline
Implementation of kubeflow deployment pipeline. This pipeline compiles and runs a new pipeline in Vertex AI that deploys a new model version onto a given endpoint.
The deployed model is tested in a final pipeline step.

start the pipeline execution with
```shell
python ./pipeline/pipeline.py
```
