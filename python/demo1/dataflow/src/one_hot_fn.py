from typing import Optional

import tensorflow as tf
from apache_beam.ml.transforms import tft
from apache_beam.ml.transforms.tft import TFTOperation, register_input_dtype
from tensorflow_transform import analyzers
from tensorflow_transform import common
from tensorflow_transform import common_types


@common.log_api_use(common.MAPPER_COLLECTION)
def one_hot(
        x: common_types.ConsistentTensorType,
        name: Optional[str] = None
) -> common_types.ConsistentTensorType:

    with tf.compat.v1.name_scope(name, 'one_hot'):
        depth = tf.cast(analyzers.max(x), tf.int32) + 1
        out = tf.one_hot(x.values, depth=depth)
        return out


@register_input_dtype(str)
class OneHot(TFTOperation):
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

        vocab = tft.ComputeAndApplyVocabulary(columns=self.columns)
        x_vocab = vocab.apply_transform(data=data,
                                        output_column_name=output_column_name)

        output = one_hot(x=x_vocab[output_column_name], name=self.name)
        return {output_column_name: output}
