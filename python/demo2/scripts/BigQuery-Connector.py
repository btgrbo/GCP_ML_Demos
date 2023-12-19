from google.cloud import bigquery

# Importing the train data set from Big Query
client = bigquery.Client(project='bt-int-ml-specialization')

job1 = client.query('SELECT * FROM `bt-int-ml-specialization.demo2.black-friday-train-new` LIMIT 1000')

result1 = job1.result()

df_train = result1.to_dataframe()

# Show the retrieved data set
#   print(df_train.head())


# Importing the test data set from Big Query
job2 = client.query('SELECT * FROM `bt-int-ml-specialization.demo2.black-friday-test-new` LIMIT 1000')

result2 = job2.result()

df_test = result2.to_dataframe()

print(df_test.head())

# GCP SDK login:
#   gcloud auth login
# IDE login:
#   gcloud auth application-default login