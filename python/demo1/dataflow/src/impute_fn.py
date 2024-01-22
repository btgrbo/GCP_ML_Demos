from typing import Optional

import tensorflow as tf
from apache_beam.ml.transforms.tft import TFTOperation, register_input_dtype
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types


@common.log_api_use(common.MAPPER_COLLECTION)
def impute_mean(
        x: common_types.ConsistentTensorType,
        name: Optional[str] = None
) -> common_types.ConsistentTensorType:

    with tf.compat.v1.name_scope(name, 'impute_mean'):
        x = tf.cast(x, tf.float32)
        nan_mask = tf.math.is_nan(x.values)
        imp_mean = analyzers.mean(x)
        new_values = tf.where(nan_mask, imp_mean, x.values)
        sparse_tensor_no_nan = tf.sparse.SparseTensor(x.indices, new_values,
                                                      x.dense_shape)
        return sparse_tensor_no_nan


@register_input_dtype(float)
class ImputeMean(TFTOperation):
    def __init__(
            self,
            columns: list[str],
            name: Optional[str] = None
    ):
        super().__init__(columns)
        self.name = name

    def apply_transform(
            self, data: common_types.TensorType,
            output_column_name: str
    ) -> dict[str, common_types.TensorType]:

        output = impute_mean(x=data, name=self.name)
        return {output_column_name: output}
