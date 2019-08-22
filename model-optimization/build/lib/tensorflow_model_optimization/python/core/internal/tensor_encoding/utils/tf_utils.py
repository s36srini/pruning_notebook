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
"""TensorFlow utilities for the `tensor_encoding` package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf


def fast_walsh_hadamard_transform(x):
  """Applies the fast Walsh-Hadamard transform to a set of vectors.

  This method uses a composition of existing TensorFlow operations to implement
  the transform.

  Args:
    x: A `Tensor`. Must be of shape `[a, b]`, where `a` can be anything (not
      necessarily known), and `b` must be a power of two, statically known.

  Returns:
    A `Tensor` of shape `[a, b]`, where `[i, :]` is the product `x[i, :]*H`,
      where `H` is the Hadamard matrix.

  Raises:
    ValueError: If the input is not rank 2 `Tensor`, and if the second dimension
      is not a power of two.
  """
  with tf.name_scope(None, 'fast_walsh_hadamard_transform'):
    # Validate input.
    x = tf.convert_to_tensor(x)
    if x.shape.ndims != 2:
      raise ValueError(
          'Number of dimensions of x must be 2. Shape of x: %s' % x.shape)
    dim = x.shape.as_list()[1]
    if not (dim and ((dim & (dim - 1)) == 0)):
      raise ValueError('The dimension of x must be a power of two. '
                       'Provided dimension is: %s' % dim)

    h_core = tf.constant([[1., 1.], [1., -1.]],
                         dtype=x.dtype,
                         name='hadamard_weights_2x2')
    permutation = tf.constant([0, 2, 1], name='hadamard_permutation')

    # A step of the fast Walsh-Hadamard algorithm.
    def _hadamard_step(x, dim):
      """A single step in the fast Walsh-Hadamard transform."""
      x = tf.reshape(x, [-1, 2])  # Reshape so that we have a matrix.
      x = tf.matmul(x, h_core)  # Multiply.
      x = tf.reshape(x, [-1, dim // 2, 2])  # Reshape to rank-3.
      x = tf.transpose(x, perm=permutation)  # Swap last two dimensions.
      return x

    # The fast Walsh-Hadamard transform.
    for _ in range(int(np.ceil(np.log2(dim)))):
      x = _hadamard_step(x, dim)
    x = tf.reshape(x, [-1, dim])
    x /= tf.sqrt(tf.cast(dim, x.dtype))  # Normalize.
    return x


def _cmwc_random_sequence(num_elements, seed):
  """Implements a version of the Complementary Multiply with Carry algorithm.

  http://en.wikipedia.org/wiki/Multiply-with-carry

  This implementation serves as a purely TensorFlow implementation of a fully
  deterministic source of pseudo-random number sequence. That is given a
  `Tensor` `seed`, this method will output a `Tensor` with `n` elements, that
  will produce the same sequence when evaluated (assuming the same value of the
  `Tensor` `seed`).

  This method is not particularly efficient, does not come with any guarantee of
  the period length, and should be replaced by appropriate alternative in
  TensorFlow 2.x. In a test in general colab runtime, it took ~0.5s to generate
  1 million values.

  Args:
    num_elements: A Python integer. The number of random values to be generated.
    seed: A scalar `Tensor` of type `tf.int64`.

  Returns:
    A `Tensor` of shape `(num_elements)` and dtype tf.float64, containing random
    values in the range `[0, 1)`.
  """
  if not isinstance(num_elements, int):
    raise TypeError('The num_elements argument must be a Python integer.')
  if num_elements <= 0:
    raise ValueError('The num_elements argument must be positive.')
  if not tf.contrib.framework.is_tensor(seed) or seed.dtype != tf.int64:
    raise TypeError('The seed argument must be a tf.int64 Tensor.')

  # For better efficiency of tf.while_loop, we generate `parallelism` random
  # sequences in parallel. The specific constant (sqrt(num_elements) / 10) is
  # hand picked after simple benchmarking for large values of num_elements.
  parallelism = int(math.ceil(math.sqrt(num_elements) / 10))
  num_iters = num_elements // parallelism + 1

  # Create constants needed for the algorithm. The constants and notation
  # follows from the above reference.
  a = tf.tile(tf.constant([3636507990], tf.int64), [parallelism])
  b = tf.tile(tf.constant([2**32], tf.int64), [parallelism])
  logb_scalar = tf.constant(32, tf.int64)
  logb = tf.tile([logb_scalar], [parallelism])
  f = tf.tile(tf.constant([0], dtype=tf.int64), [parallelism])
  bits = tf.constant(0, dtype=tf.int64, name='bits')

  # TensorArray used in tf.while_loop for efficiency.
  values = tf.TensorArray(
      dtype=tf.float64, size=num_iters, element_shape=[parallelism])
  # Iteration counter.
  num = tf.constant(0, dtype=tf.int32, name='num')
  # TensorFlow constant to be used at multiple places.
  val_53 = tf.constant(53, tf.int64, name='val_53')

  # Construct initial sequence of seeds.
  # From a single input seed, we construct multiple starting seeds for the
  # sequences to be computed in parallel.
  def next_seed_fn(i, val, q):
    val = val**7 + val**6 + 1  # PRBS7.
    q = q.write(i, val)
    return i + 1, val, q

  q = tf.TensorArray(dtype=tf.int64, size=parallelism, element_shape=())
  _, _, q = tf.while_loop(lambda i, _, __: i < parallelism,
                          next_seed_fn,
                          [tf.constant(0), seed, q])
  c = q = q.stack()

  # The random sequence generation code.
  def cmwc_step(f, bits, q, c, num, values):
    """A single step of the modified CMWC algorithm."""
    t = a * q + c
    c = b - 1 - tf.bitwise.right_shift(t, logb)
    x = q = tf.bitwise.bitwise_and(t, (b - 1))
    f = tf.bitwise.bitwise_or(tf.bitwise.left_shift(f, logb), x)
    if parallelism == 1:
      f.set_shape((1,))  # Correct for failed shape inference.
    bits += logb_scalar
    def add_val(bits, f, values, num):
      new_val = tf.cast(
          tf.bitwise.bitwise_and(f, (2**val_53 - 1)),
          dtype=tf.float64) * (1 / 2**val_53)
      values = values.write(num, new_val)
      f += tf.bitwise.right_shift(f, val_53)
      bits -= val_53
      num += 1
      return bits, f, values, num
    bits, f, values, num = tf.cond(bits >= val_53,
                                   lambda: add_val(bits, f, values, num),
                                   lambda: (bits, f, values, num))
    return f, bits, q, c, num, values

  def condition(f, bits, q, c, num, values):  # pylint: disable=unused-argument
    return num < num_iters

  _, _, _, _, _, values = tf.while_loop(
      condition,
      cmwc_step,
      [f, bits, q, c, num, values],
  )

  values = tf.reshape(values.stack(), [-1])
  # We generated parallelism * num_iters random values. Take a slice of the
  # first num_elements for the requested Tensor.
  values = values[:num_elements]
  values.set_shape((num_elements,))  # Correct for failed shape inference.
  return  values


def random_signs(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random +1/-1 values as `dtype`."""
  return tf.cast(
      tf.sign(_cmwc_random_sequence(num_elements, seed) - 0.5), dtype)


def random_floats(num_elements, seed, dtype=tf.float32):
  """Returns a Tensor of `num_elements` random values in [0, 1) as `dtype`."""
  if dtype not in [tf.float32, tf.float64]:
    raise TypeError(
        'Unsupported type: %s. Supported types are tf.float32 and '
        'tf.float64 values' % dtype)
  return tf.cast(_cmwc_random_sequence(num_elements, seed), dtype)
