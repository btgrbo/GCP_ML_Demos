# Demo 2 - Black Friday

This demo implements the machine learning solution to solve the [black friday](https://www.kaggle.com/datasets/sdolezel/black-friday) regression problem.

## Structure

### Pipeline
Implementation of kubeflow training pipeline. This pipelien compiles and runs a new pipeline in Vertex AI that trains and deploys a new model version.

start the pipeline execution with
```shell
poetry run python ./pipeline/pipeline.py
```

### Prediction
Code for the serving container. Needed to run the model in production. 
FastAPI implementation of a webservice that loads a model from GCS and listens to requests on port 8080.

To run server locally execute:
```shell
export MODEL_URI="<GCS path to model in pickle format here>"
poetry run python ./production/main.py 
```

### Training
Implementation of the action regression model. This code is supposed to be executed via pipeline.

To run the training locally:
```shell
poetry run python ./training/main.py \
    --train-file-parquet="<path to training data here>" \
    --eval-file-parquet="<path to evaluation data here>" \
    --model-file="/tmp/model.pkl" \
    --eval-output-file-parquet="/tmp/eval.parquet" \
    --train-output-file-parquet="/tmp/training.parquet" \
    --hyperparameter-tuning=true/false
```


## Deployment
New images are automatically deployed via Cloud Build with every merge in `main`. For manual deployment run `bash ./update_images.sh`