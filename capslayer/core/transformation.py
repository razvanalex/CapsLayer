# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import capslayer as cl


def transforming(inputs, num_outputs, out_caps_dims, name=None):
    """
    Args:
        inputs: A 4-D or 6-D tensor, [batch_size, num_inputs] + in_caps_dims or [batch_size, height, width, channels] + in_caps_dims.
        num_outputs: Integer, the number of output capsules.
        out_caps_dims: A list of 2 integers. The dimensions of output capsule, e.g. out_caps_dims=[4, 4].
        name: String, a name for this operation.

    Returns:
        votes: A 5-D or 7-D tensor, [batch_size, num_inputs, num_outputs] + out_caps_dims or [batch_size, height, width, channels, num_outputs] + out_caps_dims.
    """
    name = "transforming" if name is None else name
    with tf.name_scope(name) as scope:
        input_shape = cl.shape(inputs)
        prefix_shape = [1 for i in range(len(input_shape) - 3)] + input_shape[-3:-2] + [num_outputs]
        in_caps_dims = input_shape[-2:]
        if in_caps_dims[0] == out_caps_dims[1]:
            shape = prefix_shape + [out_caps_dims[0], 1, in_caps_dims[1]]
            expand_axis = -3
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[0]:
            shape = prefix_shape + [in_caps_dims[0], 1, out_caps_dims[1]]
            expand_axis = -1
            reduce_sum_axis = -3
        elif in_caps_dims[0] == out_caps_dims[0]:
            shape = prefix_shape + [1, out_caps_dims[1], in_caps_dims[1]]
            expand_axis = -2
            reduce_sum_axis = -1
        elif in_caps_dims[1] == out_caps_dims[1]:
            shape = prefix_shape + [in_caps_dims[0], out_caps_dims[0], 1]
            expand_axis = -2
            reduce_sum_axis = -3
        else:
            raise TypeError("out_caps_dims must have at least one value being the same with the in_caps_dims")
        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=expand_axis)
        transform_mat = tf.Variable(initial_value=tf.random_uniform_initializer()(shape=shape),
                                    name="transformation_matrix", shape=shape)
        votes = tf.reduce_sum(in_pose * transform_mat, axis=reduce_sum_axis)

        return votes


class Transforming(tf.keras.layers.Layer):
    def __init__(self, num_outputs, out_caps_dims, name='transforming',
                 **kwargs):
        super(Transforming, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs
        self.out_caps_dims = out_caps_dims

    def compute_shapes(self, input_shape):
        self.prefix_shape = [1 for i in range(len(input_shape) - 3)] + \
            input_shape[-3:-2] + [self.num_outputs]
        self.in_caps_dims = input_shape[-2:]

        if self.in_caps_dims[0] == self.out_caps_dims[1]:
            self.shape = self.prefix_shape + \
                [self.out_caps_dims[0], 1, self.in_caps_dims[1]]
            self.expand_axis = -3
            self.reduce_sum_axis = -1
        elif self.in_caps_dims[1] == self.out_caps_dims[0]:
            self.shape = self.prefix_shape + \
                [self.in_caps_dims[0], 1, self.out_caps_dims[1]]
            self.expand_axis = -1
            self.reduce_sum_axis = -3
        elif self.in_caps_dims[0] == self.out_caps_dims[0]:
            self.shape = self.prefix_shape + \
                [1, self.out_caps_dims[1], self.in_caps_dims[1]]
            self.expand_axis = -2
            self.reduce_sum_axis = -1
        elif self.in_caps_dims[1] == self.out_caps_dims[1]:
            self.shape = self.prefix_shape + \
                [self.in_caps_dims[0], self.out_caps_dims[0], 1]
            self.expand_axis = -2
            self.reduce_sum_axis = -3
        else:
            raise TypeError("out_caps_dims must have at least one value being the same with the in_caps_dims")

    def build(self, input_shape):
        self.compute_shapes(input_shape)

        self.transform_mat = tf.Variable(
            initial_value=tf.random_uniform_initializer()(shape=self.shape),
            name="transformation_matrix", shape=self.shape)

        super(Transforming, self).build(input_shape)

    def call(self, inputs):
        in_pose = tf.expand_dims(inputs, axis=-3)
        ones = tf.ones(shape=self.prefix_shape + [1, 1])
        in_pose = tf.expand_dims(in_pose * ones, axis=self.expand_axis)
        votes = tf.reduce_sum(in_pose * self.transform_mat, axis=self.reduce_sum_axis)

        return votes
