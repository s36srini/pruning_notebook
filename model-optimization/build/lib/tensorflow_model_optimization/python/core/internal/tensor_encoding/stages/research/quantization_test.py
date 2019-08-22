# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.stages.research import quantization
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


class PRNGUniformQuantizationEncodingStageTest(test_utils.BaseEncodingStageTest
                                              ):

  def default_encoding_stage(self):
    """See base class."""
    return quantization.PRNGUniformQuantizationEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([50], minval=-1.0, maxval=1.0)

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self._assert_is_integer_float(data.encoded_x[
        quantization.PRNGUniformQuantizationEncodingStage.ENCODED_VALUES_KEY])

  def _assert_is_integer_float(self, quantized_vals):
    """Asserts that float type values are integers."""
    assert quantized_vals.dtype == np.float32
    self.assertAllClose(quantized_vals,
                        tf.cast(tf.cast(quantized_vals, np.int32), np.float32))

  @parameterized.parameters(itertools.product([1, 2, 3, 4, 7, 8, 9, 16]))
  def test_quantization_bits_stochastic_rounding(self, bits):
    stage = quantization.PRNGUniformQuantizationEncodingStage(bits=bits)
    test_data = self.run_one_to_many_encode_decode(stage, self.default_input)
    self._assert_is_integer_float(test_data.encoded_x[
        quantization.PRNGUniformQuantizationEncodingStage.ENCODED_VALUES_KEY])
    # For stochastic rounding, the potential error incurred by quantization
    # is bounded by the range of the input values divided by the number of
    # quantization buckets.
    self.assertAllClose(
        test_data.x, test_data.decoded_x, rtol=0.0, atol=2 / (2**bits - 1))

  def test_quantization_empirically_unbiased(self):
    # Tests that the quantization "seems" to be unbiased.
    # Executing the encoding and decoding many times, the average error should
    # be a lot larger than the error of average decoded value.
    x = tf.constant(np.random.rand((50)).astype(np.float32))
    stage = quantization.PRNGUniformQuantizationEncodingStage(bits=2)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data_list = [self.evaluate_test_data(test_data) for _ in range(200)]

    norm_errors = []
    errors = []
    for data in test_data_list:
      norm_errors.append(np.linalg.norm(data.x - data.decoded_x))
      errors.append(data.x - data.decoded_x)
    mean_of_errors = np.mean(norm_errors)
    error_of_mean = np.linalg.norm(np.mean(errors, axis=0))
    self.assertGreater(mean_of_errors, error_of_mean * 10)

  @parameterized.parameters(
      itertools.product([tf.float32, tf.float64], [tf.float32, tf.float64]))
  def test_input_types(self, x_dtype, min_max_dtype):
    # Tests combinations of input dtypes.
    stage = quantization.PRNGUniformQuantizationEncodingStage(bits=8)
    x = tf.random.normal([50], dtype=x_dtype)
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = test_utils.TestData(x, encoded_x, decoded_x)
    test_data = self.evaluate_test_data(test_data)

    self.assertAllGreaterEqual(test_data.decoded_x, np.amin(test_data.x))
    self.assertAllLessEqual(test_data.decoded_x, np.amax(test_data.x))

  def test_all_zero_input_works(self):
    # Tests that encoding does not blow up with all-zero input. With
    # min_max=None, the derived min and max are identical, thus potential for
    # division by zero.
    stage = quantization.PRNGUniformQuantizationEncodingStage(bits=8)
    test_data = self.run_one_to_many_encode_decode(
        stage, lambda: tf.zeros([50]))
    self.assertAllEqual(np.zeros((50)).astype(np.float32), test_data.decoded_x)

  @parameterized.parameters([0, 17, -1, 1.5])
  def test_bits_out_of_range_raises(self, bits):
    with self.assertRaisesRegexp(ValueError, 'integer between 1 and 16'):
      quantization.PRNGUniformQuantizationEncodingStage(bits=bits)


if __name__ == '__main__':
  tf.test.main()
