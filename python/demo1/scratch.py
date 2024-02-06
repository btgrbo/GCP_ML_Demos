import tensorflow as tf

# Path to your TFRecord file
tfrecord_file_path = 'gs://bt-int-ml-specialization_dataflow_demo1/TFRecords/run_2024-01-31T16:17:33.578849-00000-of-00001.tfrecord'

# Create a TFRecordDataset to read the TFRecord file
raw_dataset = tf.data.TFRecordDataset([tfrecord_file_path])

# Read the first raw record
for raw_record in raw_dataset.take(1):
    # Parse the raw_record as a tf.train.Example (assuming it was saved in this format)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    print("Features included in the TFRecord:")
    for key in example.features.feature.keys():
        print(key)

    # Break after the first record to just see the feature names
    break
