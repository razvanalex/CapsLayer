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

import numpy as np
import capslayer as cl
import tensorflow as tf

from capslayer.core import Routing
from capslayer.core import Transforming


class PrimaryCapsule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, out_caps_dims, strides=1,
                 method=None, name="primary_capsule", **kwargs):
        super(PrimaryCapsule, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_caps_dims = out_caps_dims
        self.method = method

        self.channels = filters * np.prod(out_caps_dims)
        self.channels = self.channels + \
            self.filters if self.method == "logistic" else self.channels

        self.conv2d = tf.keras.layers.Conv2D(self.channels,
                                             kernel_size=self.kernel_size,
                                             strides=self.strides,
                                             activation=None)

    def build(self, input_shape):
        super(PrimaryCapsule, self).build(input_shape)

    def call(self, inputs):
        pose = self.conv2d(inputs)
        shape = cl.shape(pose, name="get_pose_shape")
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        shape = [batch_size, height, width, self.filters] + self.out_caps_dims

        if self.method == 'logistic':
            # logistic activation unit
            num_or_size_splits = [self.channels - self.filters, self.filters]
            pose, activation_logit = tf.split(pose, num_or_size_splits,
                                              axis=-1)
            pose = tf.reshape(pose, shape=shape)
            activation = tf.sigmoid(activation_logit)
        elif self.method == 'norm' or self.method is None:
            pose = tf.reshape(pose, shape=shape)
            squash_on = -2 if self.out_caps_dims[-1] == 1 else [-2, -1]
            pose = cl.ops.squash(pose, axis=squash_on)
            activation = cl.norm(pose, axis=(-2, -1))
        activation = tf.clip_by_value(activation, 1e-20, 1. - 1e-20)

        return (pose, activation)


class CapsuleConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, out_caps_dims, kernel_size,
                 strides=1, padding="valid", routing_method="EMRouting",
                 routing_iter=3, name="caps_1d", **kwargs):
        super(CapsuleConv1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_caps_dims = out_caps_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.routing_method = routing_method
        self.routing_iter = routing_iter

        self.conv2d = CapsuleConv2D(filters=filters,
                                    out_caps_dims=out_caps_dims,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    routing_method=routing_method,
                                    routing_iter=routing_iter)

    def build(self, input_shape):
        input_shape, activation_shape = input_shape
        input_rank = len(input_shape)
        activation_rank = len(activation_shape)
        if input_rank != 5:
            raise ValueError('Inputs to `conv1d` should have rank 5. Received input rank:',
                             str(input_rank))
        if activation_rank != 3:
            raise ValueError('Activation to `conv1d` should have rank 3. Received input shape:',
                             str(activation_rank))

        if isinstance(self.kernel_size, int):
            self.kernel_size = [1, self.kernel_size]
        elif isinstance(self.kernel_size, (list, tuple)) and len(self.kernel_size) == 1:
            self.kernel_size = [1, self.kernel_size[0]]
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 2 integers. Received:',
                             str(self.kernel_size))

        if isinstance(self.strides, int):
            self.strides = [1, self.strides]
        elif isinstance(self.strides, (list, tuple)) and len(self.strides) == 1:
            self.strides = [1, self.strides[0]]
        else:
            raise ValueError('"stride" should be an integer or tuple/list of a single integer. Received:',
                             str(self.strides))

        if not isinstance(self.out_caps_dims, (list, tuple)) or len(self.out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:',
                             str(self.out_caps_dims))
        elif isinstance(self.out_caps_dims, tuple):
            self.out_caps_dims = list(self.out_caps_dims)

        super(CapsuleConv1D, self).build(input_shape)

    def call(self, inputs):
        inputs, activation = inputs
        inputs = tf.expand_dims(inputs, axis=1)
        activation = tf.expand_dims(activation, axis=1)
        pose, activation = self.conv2d(inputs, activation)
        pose = tf.squeeze(pose, axis=1)
        activation = tf.squeeze(activation, axis=1)

        return pose, activation


class CapsuleConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, out_caps_dims, kernel_size,
                 strides=1, padding="valid", routing_method="EMRouting",
                 routing_iter=3, name="caps_2d", **kwargs):
        super(CapsuleConv2D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_caps_dims = out_caps_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.routing_method = routing_method
        self.routing_iter = routing_iter

        self.transforming = Transforming(filters, out_caps_dims)
        self.routing = Routing(routing_method, routing_iter)

    def build(self, input_shape):
        input_shape, activation_shape = input_shape
        input_rank = len(input_shape)
        activation_rank = len(activation_shape)
        if not input_rank == 6:
            raise ValueError('Inputs to `conv2d` should have rank 6. Received inputs rank:',
                             str(input_rank))
        if not activation_rank == 4:
            raise ValueError('Activation to `conv2d` should have rank 4. Received activation rank:',
                             str(activation_rank))

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size,
                                self.kernel_size, input_shape[3]]
        elif isinstance(self.kernel_size, (list, tuple)) and len(self.kernel_size) == 2:
            self.kernel_size = [self.kernel_size[0],
                                self.kernel_size[1], input_shape[3]]
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 2 integers. Received:',
                             str(self.kernel_size))

        if isinstance(self.strides, int):
            self.strides = [self.strides, self.strides, 1]
        elif isinstance(self.strides, (list, tuple)) and len(self.strides) == 2:
            self.strides = [self.strides[0], self.strides[1], 1]
        else:
            raise ValueError('"strides" should be an integer or tuple/list of 2 integers. Received:',
                             str(self.kernel_size))

        if not isinstance(self.out_caps_dims, (list, tuple)) or len(self.out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:',
                             str(self.out_caps_dims))
        elif isinstance(self.out_caps_dims, tuple):
            self.out_caps_dims = list(self.out_caps_dims)

        super(CapsuleConv2D, self).build(input_shape)

    def call(self, inputs):
        inputs, activation = inputs

        # 1. space to batch
        # patching everything into [batch_size, out_height, out_width, in_channels] + in_caps_dims (batched)
        # and [batch_size, out_height, out_width, in_channels] (activation).
        batched = cl.space_to_batch_nd(inputs, self.kernel_size, self.strides)
        activation = cl.space_to_batch_nd(activation,
                                          self.kernel_size, self.strides)

        # 2. transforming
        # transforming to [batch_size, out_height, out_width, in_channels, out_channels/filters] + out_caps_dims
        vote = self.transforming(batched)

        # 3. routing
        pose, activation = self.routing(vote, activation)

        return pose, activation


class CapsuleConv3D(tf.keras.layers.Layer):
    def __init__(self, filters, out_caps_dims, kernel_size,
                 strides=1, padding="valid", routing_method="EMRouting",
                 routing_iter=3, name="caps_3d", **kwargs):
        super(CapsuleConv3D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.out_caps_dims = out_caps_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.routing_method = routing_method
        self.routing_iter = routing_iter

        self.transforming = Transforming(filters, out_caps_dims)
        self.routing = Routing(routing_method, routing_iter)

    def build(self, input_shape):
        input_shape, activation_shape = input_shape
        input_rank = len(input_shape)
        activation_rank = len(activation_shape)
        if input_rank != 7:
            raise ValueError('Inputs to `conv3d` should have rank 7. Received input rank:',
                             str(input_rank))
        if activation_rank != 5:
            raise ValueError('Activation to `conv3d` should have rank 5. Received input shape:',
                             str(activation_rank))

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size,
                                self.kernel_size, self.kernel_size]
        elif isinstance(self.kernel_size, (list, tuple)) and len(self.kernel_size) == 3:
            self.kernel_size = self.kernel_size
        else:
            raise ValueError('"kernel_size" should be an integer or tuple/list of 3 integers. Received:',
                             str(self.kernel_size))

        if isinstance(self.strides, int):
            self.strides = [self.strides, self.strides, self.strides]
        elif isinstance(self.strides, (list, tuple)) and len(self.strides) == 3:
            self.strides = self.strides
        else:
            raise ValueError('"strides" should be an integer or tuple/list of 3 integers. Received:',
                             str(self.strides))

        if not isinstance(self.out_caps_dims, (list, tuple)) or len(self.out_caps_dims) != 2:
            raise ValueError('"out_caps_dims" should be a tuple/list of 2 integers. Received:',
                             str(self.out_caps_dims))
        elif isinstance(self.out_caps_dims, tuple):
            self.out_caps_dims = list(self.out_caps_dims)

        super(CapsuleConv3D, self).build(input_shape)

    def call(self, inputs):
        inputs, activation = inputs
    
        # 1. space to batch
        batched = cl.space_to_batch_nd(inputs, self.kernel_size, self.strides)
        activation = cl.space_to_batch_nd(activation,
                                          self.kernel_size, self.strides)

        # 2. transforming
        vote = self.transforming(batched)

        # 3. routing
        pose, activation = self.routing(vote, activation)

        return pose, activation


class CapsuleDense(tf.keras.layers.Layer):
    def __init__(self, num_outputs, out_caps_dims,
                 routing_method="EMRouting", routing_iter=3,
                 coordinate_addition=False, name="caps_dense",
                 **kwargs):
        super(CapsuleDense, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs
        self.out_caps_dims = out_caps_dims
        self.routing_method = routing_method
        self.routing_iter = routing_iter
        self.coordinate_addition = coordinate_addition

        self.transforming = Transforming(self.num_outputs, self.out_caps_dims)
        self.routing = Routing(routing_method, routing_iter)

    def build(self, input_shape):
        # input_shape, activation_shape = input_shape
        # self.transforming.build(input_shape)

        # vote_shape = input_shape[:-3] + [self.num_outputs] + self.out_caps_dims
        # self.routing.build(vote_shape)

        super(CapsuleDense, self).build(input_shape)

    def call(self, inputs):
        inputs, activation = inputs

        if self.coordinate_addition and len(inputs.shape) == 6 and len(activation.shape) == 4:
            vote = self.transforming(inputs)

            batch_size, in_height, in_width, in_channels, _, out_caps_height, out_caps_width = cl.shape(
                vote)
            num_inputs = in_height * in_width * in_channels

            zeros = np.zeros((in_height, out_caps_width - 1))
            coord_offset_h = ((np.arange(in_height) + 0.5) /
                              in_height).reshape([in_height, 1])
            coord_offset_h = np.concatenate([zeros, coord_offset_h], axis=-1)
            zeros = np.zeros((out_caps_height - 1, out_caps_width))
            coord_offset_h = np.stack([np.concatenate(
                [coord_offset_h[i:(i + 1), :], zeros], axis=0) for i in range(in_height)], axis=0)
            coord_offset_h = coord_offset_h.reshape(
                (1, in_height, 1, 1, 1, out_caps_height, out_caps_width))

            zeros = np.zeros((1, in_width))
            coord_offset_w = ((np.arange(in_width) + 0.5) /
                              in_width).reshape([1, in_width])
            coord_offset_w = np.concatenate(
                [zeros, coord_offset_w, zeros, zeros], axis=0)
            zeros = np.zeros((out_caps_height, out_caps_width - 1))
            coord_offset_w = np.stack([np.concatenate(
                [zeros, coord_offset_w[:, i:(i + 1)]], axis=1) for i in range(in_width)], axis=0)
            coord_offset_w = coord_offset_w.reshape(
                (1, 1, in_width, 1, 1, out_caps_height, out_caps_width))

            vote = vote + tf.constant(coord_offset_h +
                                      coord_offset_w, dtype=tf.float32)

            vote = tf.reshape(
                vote, shape=[batch_size, num_inputs, self.num_outputs] + self.out_caps_dims)
            activation = tf.reshape(activation, shape=[batch_size, num_inputs])

        elif len(inputs.shape) == 4 and len(activation.shape) == 2:
            vote = self.transforming(inputs)

        else:
            raise TypeError("Wrong rank for inputs or activation")

        pose, activation = self.routing(vote, activation)

        assert len(pose.shape) == 4
        assert len(activation.shape) == 2
        
        return (pose, activation)
