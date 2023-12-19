from kfp import dsl
from google.cloud import bigquery
import pandas as pd

@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"])
def load_and_save_data(
        train_data: dsl.Output[dsl.Dataset],
        test_data: dsl.Output[dsl.Dataset],
        train_query: str,
        test_query: str,
):
    client = bigquery.Client()

    # Load data from BigQuery view based on the SQL query
    train_job = client.query(train_query)
    train_result = train_job.result()
    df_train = train_result.to_dataframe()

    test_job = client.query(test_query)
    test_result = test_job.result()
    df_test = test_result.to_dataframe()

    def save_data(df, dataset):
        dataset.from_dataframe(df)

    save_data(df_train, train_data)
    save_data(df_test, test_data)
