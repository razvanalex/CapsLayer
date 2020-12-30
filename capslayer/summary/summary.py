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


def image(name,
          tensor,
          step=None,
          verbose=True,
          max_outputs=3):
    if verbose:
        if step is not None:
            tf.summary.image(name, tensor, step, max_outputs)
        else:
            tf.summary.image(name, tensor, max_outputs=max_outputs)
    else:
        pass


def scalar(name, tensor, step=None, verbose=False):
    if verbose:
        if step is not None:
            tf.summary.scalar(name, tensor, step)
        else:
            tf.summary.scalar(name, tensor)
    else:
        pass


def histogram(name, values, step=None, verbose=False):
    if verbose:
        if step is not None:
            tf.summary.histogram(name, values, step)
        else:
            tf.summary.histogram(name, values)
    else:
        pass


def tensor_stats(name, tensor, step=None, verbose=True):
    """
    Args:
        tensor: A non-scalar tensor.
    """
    if verbose:
        with tf.name_scope(name):
            mean = tf.reduce_mean(tensor)
            tf.summary.scalar('mean', mean, step)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
            tf.summary.scalar('stddev', stddev, step)
            tf.summary.scalar('max', tf.reduce_max(tensor), step)
            tf.summary.scalar('min', tf.reduce_min(tensor), step)
            tf.summary.histogram('histogram', tensor, step)
    else:
        pass
