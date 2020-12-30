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

"""Input utility functions for reading Cifar10 dataset.

Handles reading from Cifar10 dataset saved in binary original format. Scales and
normalizes the images as the preprocessing step. It can distort the images by
random cropping and contrast adjusting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from capslayer.data.datasets.cifar100.writer import tfrecord_runner


def parse_fun(serialized_example):
    """ Data parsing function.
    """
    features = tf.io.parse_single_example(serialized_example,
                                       features={'image': tf.io.FixedLenFeature([], tf.string),
                                                 'label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.io.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, shape=[32 * 32 * 3])
    image.set_shape([32 * 32 * 3])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)
    features = {'images': image, 'labels': label}
    return(features)


class DataLoader(object):
    """ Data Loader.
    """
    def __init__(self, path=None,
                 num_works=1,
                 splitting="TVT",
                 one_hot=False,
                 name="create_inputs"):

        if path is None or not os.path.exists(path):
            tfrecord_runner()

        self.path = path
        self.name = name

    def __call__(self, batch_size, mode):
        """
        Args:
            batch_size: Integer.
            mode: Running phase, one of "train", "test" or "eval"(only if splitting='TVT').
        """
        with tf.name_scope(self.name):
            mode = mode.lower()
            modes = ["train", "test", "eval"]
            if mode not in modes:
                raise "mode not found! supported modes are " + modes

            filenames = [os.path.join(self.path, '%s_cifar100.tfrecord' % mode)]
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(parse_fun)
            dataset = dataset.batch(batch_size)

            if mode == "train":
                dataset = dataset.shuffle(buffer_size=50000)
                dataset = dataset.repeat()
            elif mode == "eval":
                dataset = dataset.repeat(1)
            elif mode == "test":
                dataset = dataset.repeat(1)

            return dataset
