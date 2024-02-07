import tensorflow as tf

# Path to your TFRecord file
#tfrecord_file_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-01-31T16:17:33.578849-00000-of-00001.tfrecord'\
tfrecord_file_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-02-06T16:50:48.097865-00000-of-00001.tfrecord'


# Create a TFRecordDataset to read the TFRecord file
raw_dataset = tf.data.TFRecordDataset([tfrecord_file_path])

# Define the feature description dictionary
# This is needed to parse the tf.train.Example messages
feature_description = {
    #'fare': tf.io.FixedLenFeature([], tf.float32),
    #'trip_miles': tf.io.FixedLenFeature([], tf.float32),
    #'trip_seconds': tf.io.FixedLenFeature([], tf.float32),
    #'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
    #'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
    #'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
    #'pickup_longitude': tf.io.FixedLenFeature([], tf.float32),
    'start_hour': tf.io.VarLenFeature(tf.float32),
    #'start_month': tf.io.VarLenFeature(tf.float32),
    #'start_date': tf.io.VarLenFeature(tf.float32),
    #'day_of_week': tf.io.VarLenFeature(tf.float32)
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
