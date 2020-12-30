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
                 routing_method='DynamicRouting'):
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
        self.num_label = num_label
        self.routing_method = routing_method
        self.num_outputs = height * width * channels
        
        out_caps_dims = [4, 1]

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=5,
                                            strides=2,
                                            padding='VALID',
                                            activation=tf.nn.relu,
                                            name="Conv1_layer")

        self.primaryCaps = cl.keras.layers.PrimaryCapsule(filters=32,
                                                          kernel_size=1,
                                                          strides=1,
                                                          out_caps_dims=out_caps_dims,
                                                          method="logistic",
                                                          name="PrimaryCaps_layer")

        self.capsConv2D_1 = cl.keras.layers.CapsuleConv2D(filters=32,
                                                          out_caps_dims=out_caps_dims,
                                                          kernel_size=(3, 3),
                                                          strides=(2, 2),
                                                          routing_method=routing_method,
                                                          name="ConvCaps1_layer")

        self.capsConv2D_2 = cl.keras.layers.CapsuleConv2D(filters=32,
                                                          out_caps_dims=out_caps_dims,
                                                          kernel_size=(3, 3),
                                                          strides=(1, 1),
                                                          routing_method=routing_method,
                                                          name="ConvCaps2_layer")

        self.denseCaps = cl.keras.layers.CapsuleDense(num_outputs=num_label,
                                                      out_caps_dims=out_caps_dims,
                                                      routing_method=routing_method,
                                                      coordinate_addition=True,
                                                      name="ClassCaps_layer")

        self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu,
                                         name='fc_1')
        self.fc2 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu,
                                         name='fc_2')
        self.recon_img = tf.keras.layers.Dense(units=self.num_outputs,
                                               activation=tf.sigmoid, name='recon_img')

    def forward(self, inputs):
        """ Setup capsule network.
        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels] or [batch_size, height * width * channels].

        Returns:
            poses: Tensor with shape [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """
        probs_list = []

        x = inputs
        x = tf.reshape(x, shape=[-1, self.height, self.width, self.channels])
        x = self.conv1(x)

        x, activation = self.primaryCaps(x)
        probs_list.append(tf.reduce_mean(activation))

        x, activation = self.capsConv2D_1((x, activation))
        probs_list.append(tf.reduce_mean(activation))

        x, activation = self.capsConv2D_2((x, activation))
        probs_list.append(tf.reduce_mean(activation))

        poses, probs = self.denseCaps((x, activation))
        probs_list.append(tf.reduce_mean(probs))

        tf.summary.scalar("probs", tf.reduce_mean(probs_list))

        return poses, probs

    def decoder(self, poses, labels):
        """ Decode the poses
        Args:
            poses: [batch_size, num_label, 16, 1].
            labels: Tensor or array with shape [batch_size].
        Returns:
            img: batch_size reconstructed images.
        """
        # Decoder structure
        # Reconstructe the inputs with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
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
        cl.summary.histogram('activation', tf.nn.softmax(
            probs, 1), verbose=cfg.summary_verbose)

        logits_idx = tf.cast(
            tf.argmax(cl.softmax(probs, axis=1), axis=1), tf.int32)
        correct_prediction = tf.equal(tf.cast(labels, tf.int32), logits_idx)
        correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.reduce_mean(
            correct / tf.cast(tf.shape(probs)[0], tf.float32))

        cl.summary.scalar('accuracy', accuracy, verbose=cfg.summary_verbose)

        return accuracy

    def _loss(self, inputs, probs, recon_img, global_step):
        # The reconstruction loss
        recon_img = tf.reshape(recon_img,
                               shape=(-1, self.height * self.width * self.channels))
        original_img = tf.reshape(inputs,
                                  shape=(-1, self.height * self.width * self.channels))
        squared = tf.square(recon_img - original_img)
        reconstruction_err = tf.reduce_mean(squared)
        
        cl.summary.scalar('reconstruction_loss',
                          reconstruction_err, verbose=cfg.summary_verbose)

        # Spread loss
        initial_margin = 0.2
        max_margin = 0.9
        interstep = 8000
        margin = (global_step / interstep) * 0.1 + initial_margin
        margin = tf.cast(tf.minimum(margin, max_margin), tf.float32)
        cl.summary.scalar('margin', tf.reduce_mean(
            margin), verbose=cfg.summary_verbose)
        
        spread_loss = cl.losses.spread_loss(logits=probs,
                                            labels=tf.squeeze(
                                                self.labels_one_hot, axis=(2, 3)),
                                            margin=margin)
        
        cl.summary.scalar('spread_loss', spread_loss,
                          verbose=cfg.summary_verbose)

        # Total loss
        total_loss = spread_loss + cfg.regularization_scale * reconstruction_err

        cl.summary.scalar('total_loss', total_loss,
                          verbose=cfg.summary_verbose)

        return total_loss

    def train_step(self, inputs, labels, global_step=None):
        optimizer = tf.keras.optimizers.Adam()

        with tf.GradientTape() as tape:
            poses, probs = self.forward(inputs)
            recon_img = self.decoder(poses, labels)
            total_loss = self._loss(inputs, probs, recon_img, global_step)

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

    def call(self, inputs, labels, global_step=None):
        return self.train_step(inputs, labels, global_step)
