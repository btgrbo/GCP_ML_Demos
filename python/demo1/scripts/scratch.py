import tensorflow as tf
import json

# Path to your TFRecord file
#tfrecord_file_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-01-31T16:17:33.578849-00000-of-00001.tfrecord'\
tfrecord_file_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-02-06T16:50:48.097865-00000-of-00001.tfrecord'


# Create a TFRecordDataset to read the TFRecord file
raw_dataset = tf.data.TFRecordDataset([tfrecord_file_path])

# Define the feature description dictionary
# This is needed to parse the tf.train.Example messages
feature_description = {
    'fare': tf.io.FixedLenFeature([], tf.float32),
    'trip_miles': tf.io.FixedLenFeature([], tf.float32),
    'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
    'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
    'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
    'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
    'pickup_longitude': tf.io.FixedLenFeature([], tf.float32),
    'start_hour': tf.io.VarLenFeature(tf.float32),
    'start_month': tf.io.VarLenFeature(tf.float32),
    'start_date': tf.io.VarLenFeature(tf.float32),
    'day_of_week': tf.io.VarLenFeature(tf.float32)
}

# Function to parse a single example
def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the feature description
    return tf.io.parse_single_example(example_proto, feature_description)

# Parse the first example of the dataset
for raw_record in raw_dataset.take(1):
    example = _parse_function(raw_record)
    print("Features included in the TFRecord and one example value for each:")
    for key, value in example.items():
        # Decode the value if it's a byte string
        if value.dtype == tf.string:
            # For string features, you might need to decode the byte string
            value = value._numpy().decode('utf-8')
        else:
            # Convert tensor to numpy array and get the first element for non-string features
            value = value._numpy()
        print(f"{key}: {value}")



## all records



# Parse each example in the dataset
parsed_dataset = raw_dataset.map(_parse_function)

# Iterate through the entire dataset and print features for each record
for parsed_record in parsed_dataset:
    print("Features in record:")
    for key, value in parsed_record.items():
        if value.dtype == tf.string:
            # Decode bytes to string
            print_value = value._numpy().decode('utf-8')
        else:
            # Convert tensor to numpy array for printing
            print_value = value._numpy()
        print(f"{key}: {print_value}")

from datetime import datetime
import pytz

tt = "2015-12-06 17:15:00.000000 UTC"
timestamp_format = '%Y-%m-%d %H:%M:%S.%f'
# Extract the datetime string without the ' UTC' at the end
datetime_str = tt.rsplit(' ', 1)[0]
# Parse the timestamp string to a datetime object
parsed_timestamp = datetime.strptime(datetime_str, timestamp_format)
# Since the timestamp is in UTC, attach the UTC timezone to make it timezone-aware
timestamp=parsed_timestamp.replace(tzinfo=pytz.utc)

day_of_week = timestamp.weekday()
month = timestamp.month
date = timestamp.day
hour = timestamp.hour



import tensorflow as tf

model = tf.saved_model.load('gs://bt-int-ml-specialization-ml-demo1/1/model')
print(list(model.signatures.keys()))  # List signature keys

model = tf.saved_model.load('./python/demo1/dataflow/model')
print(list(model.signatures.keys()))
signature = model.signatures['serving_default']
print(signature.structured_input_signature)
print(signature.structured_outputs)


import numpy as np

# Assuming 'data_dict' is your input dictionary
data_dict = {
    'day_of_week': np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32),
    'dropoff_latitude': np.array([0.48586452], dtype=np.float32),
    'dropoff_longitude': np.array([1.205355], dtype=np.float32),
    'pickup_latitude': np.array([5.449345], dtype=np.float32),
    'pickup_longitude': np.array([-1.8299297], dtype=np.float32),
    'start_date': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    'start_hour': np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    'start_month': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], dtype=np.float32),
    'trip_miles': np.array([-1.256183], dtype=np.float32),
    'trip_seconds': np.array([-1.3809716], dtype=np.float32)
}

# Specify the order of features
feature_order = ['start_month', 'start_date', 'day_of_week', 'start_hour', 'trip_miles', 'trip_seconds', 'dropoff_longitude', 'dropoff_latitude', 'pickup_longitude', 'pickup_latitude']

# Create the concatenated array based on the specified order
concatenated_array = np.concatenate([data_dict[feature].flatten() for feature in feature_order])

# Ensure the concatenated array is 2D with shape (1, N)
final_array = concatenated_array.reshape(1, -1)

# Convert the NumPy array to a list
dense_input_list = final_array.tolist()[0]

# Prepare the JSON payload
json_payload = {"instances": [{"dense_input": dense_input_list}]}

# Convert the payload to a JSON string for serving
json_str = json.dumps(json_payload)



################

from google.cloud import bigquery

client = bigquery.Client()
table_id = "bt-int-ml-specialization.demo1.taxi_trips_eval_dense_input_3"

row =     {
        "dense_input": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.2561830282211304, -1.3809715509414673, 1.2053550481796265, 0.485864520072937, -1.8299297094345093, 5.449345111846924],
        "fare": 1
    }

rows_to_insert = [row for rows in range(1000)]

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors == []:
    print("New rows have been added.")
else:
    print("Encountered errors while inserting rows: {}".format(errors))


import json

# Specify the filename
filename = './eval_data.jsonl'

# Open the file in write mode
with open(filename, 'w') as file:
    for item in rows_to_insert:
        # Convert dictionary to JSON string and write to file
        json_str = json.dumps(item)
        file.write(json_str + '\n')
