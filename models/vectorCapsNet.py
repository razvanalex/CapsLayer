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

from config import cfg


class CapsNet(tf.keras.Model):
    def __init__(self, height=28, width=28, channels=1, num_label=10,
                 routing_method="DynamicRouting"):
        '''
        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        '''
        super(CapsNet, self).__init__()

        self.height = height
        self.width = width
        self.channels = channels
        self.num_outputs = height * width * channels
        self.num_label = num_label

        self.conv1 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=9,
                                            strides=1,
                                            padding='valid',
                                            activation=tf.nn.relu)

        self.primaryCaps = cl.keras.layers.PrimaryCapsule(filters=32,
                                                          kernel_size=9,
                                                          strides=2,
                                                          out_caps_dims=[8, 1],
                                                          method="norm")

        self.denseCaps = cl.keras.layers.CapsuleDense(num_outputs=num_label,
                                                      out_caps_dims=[16, 1],
                                                      routing_method=routing_method)

        self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='fc_1')
        self.fc2 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, name='fc_2')
        self.recon_img = tf.keras.layers.Dense(units=self.num_outputs,
                                               activation=tf.sigmoid, name='recon_img')

    def forward(self, inputs):
        """ Computes the probs and poses
        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels] or [batch_size, height * width * channels].
        Returns:
            poses: [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """
        x = inputs
        x = tf.reshape(x, shape=[-1, self.height, self.width, self.channels])
        x = self.conv1(x)

        x, activation = self.primaryCaps(x)

        num_inputs = np.prod(cl.shape(x)[1:4])
        x = tf.reshape(x, shape=[-1, num_inputs, 8, 1])
        activation = tf.reshape(activation, shape=[-1, num_inputs])
        poses, probs = self.denseCaps(x, activation)

        cl.summary.histogram('activation', probs, verbose=cfg.summary_verbose)

        return poses, probs

    def decoder(self, poses, labels):
        """ Decode the poses
        Args:
            poses: [batch_size, num_label, 16, 1].
            labels: Tensor or array with shape [batch_size].
        Returns:
            img: batch_size reconstructed images.
        """
        labels = tf.one_hot(labels, depth=self.num_label,
                            axis=-1, dtype=tf.float32)
        self.labels_one_hot = tf.reshape(labels, (-1, self.num_label, 1, 1))

        masked_caps = tf.multiply(poses, self.labels_one_hot)

        num_inputs = np.prod(masked_caps.get_shape().as_list()[1:])
        active_caps = tf.reshape(masked_caps, shape=(-1, num_inputs))

        x = self.fc1(active_caps)
        x = self.fc2(x)
        img = self.recon_img(x)
        img = tf.reshape(img,
                         shape=[-1, self.height, self.width, self.channels])

        cl.summary.image('reconstruction_img', img,
                         verbose=cfg.summary_verbose)

        return img

    def accuracy(self, probs, labels):
        logits_idx = tf.cast(
            tf.argmax(cl.softmax(probs, axis=1), axis=1), tf.int32)
        correct_prediction = tf.equal(tf.cast(labels, tf.int32), logits_idx)
        correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(
            correct / tf.cast(tf.shape(probs)[0], tf.float32))

        cl.summary.scalar('accuracy', accuracy, verbose=cfg.summary_verbose)

        return accuracy

    def _loss(self, inputs, probs, recon_img):
        # 1. Margin loss
        margin_loss = cl.losses.margin_loss(logits=probs,
                                            labels=tf.squeeze(self.labels_one_hot,
                                                              axis=(2, 3)))

        cl.summary.scalar('margin_loss', margin_loss,
                          verbose=cfg.summary_verbose)

        # 2. The reconstruction loss
        recon_img = tf.reshape(recon_img,
                               shape=(-1, self.height * self.width * self.channels))
        original_img = tf.reshape(inputs,
                                  shape=(-1, self.height * self.width * self.channels))
        squared = tf.square(recon_img - original_img)
        reconstruction_err = tf.reduce_mean(squared)

        cl.summary.scalar('reconstruction_loss', reconstruction_err,
                          verbose=cfg.summary_verbose)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        total_loss = margin_loss + cfg.regularization_scale * reconstruction_err

        cl.summary.scalar('total_loss', total_loss,
                          verbose=cfg.summary_verbose)

        return total_loss

    def train_step(self, inputs, labels):
        optimizer = tf.keras.optimizers.Adam()

        with tf.GradientTape() as tape:
            poses, probs = self.forward(inputs)
            recon_img = self.decoder(poses, labels)
            total_loss = self._loss(inputs, probs, recon_img)

        grads = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        train_acc = self.accuracy(probs, labels)

        return (total_loss, train_acc, probs)

    def eval(self, inputs, labels=None):
        _, probs = self.forward(inputs)
        logits_idx = tf.cast(tf.argmax(cl.softmax(probs, axis=1), axis=1),
                                tf.int32)

        if labels is None:
            return probs, logits_idx, None

        return probs, logits_idx, self.accuracy(probs, labels)

    def call(self, inputs, labels):
        return self.train_step(inputs, labels)
