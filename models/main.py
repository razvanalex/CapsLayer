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

import os
import time
import numpy as np
import tensorflow as tf
from importlib import import_module

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import cfg
from capslayer.plotlib import plot_activation


def save_to(is_training):
    os.makedirs(os.path.join(cfg.results_dir, "activations"), exist_ok=True)
    os.makedirs(os.path.join(cfg.results_dir, "timelines"), exist_ok=True)

    if is_training:
        loss = os.path.join(cfg.results_dir, 'loss.csv')
        train_acc = os.path.join(cfg.results_dir, 'train_acc.csv')
        val_acc = os.path.join(cfg.results_dir, 'val_acc.csv')

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        fd = {"train_acc": fd_train_acc,
              "loss": fd_loss,
              "val_acc": fd_val_acc}
    else:
        test_acc = os.path.join(cfg.results_dir, 'test_acc.csv')
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        fd = {"test_acc": fd_test_acc}

    return(fd)


def train(model, data_loader):
    checkpoint_path = os.path.join(cfg.logdir, 'model_-{epoch:04d}.ckpt')

    # Setting up the dataloader
    training_iterator = data_loader(cfg.batch_size, mode="train")
    validation_iterator = data_loader(cfg.batch_size, mode="eval")

    # Creating files, saver and summary writer to save training results
    fd = save_to(is_training=True)
    summary_writer = tf.summary.create_file_writer(cfg.logdir)
    print("\nNote: all of results will be saved to directory: " + cfg.results_dir)

    # Compute the cardinality of the dataset
    train_ds_len = 0
    for data in training_iterator:
        train_ds_len = train_ds_len + 1

    val_ds_len = 0
    for _ in validation_iterator:
        val_ds_len = val_ds_len + 1

    # Init the model and show the summary
    data = next(iter(training_iterator))
    model(data['images'], data['labels'], 1)
    print(model.summary())

    # Train the model
    with summary_writer.as_default():
        loss_val_avg = []
        train_acc_avg = []
        for step in range(1, cfg.num_epochs):
            start_time = time.time()

            # Initialize progress bars
            progbar_train = tf.keras.utils.Progbar(train_ds_len)
            progbar_val = tf.keras.utils.Progbar(val_ds_len)

            tf.summary.experimental.set_step(step)

            # Train
            loss_val_step = []
            train_acc_step = []
            for b_id, data in enumerate(training_iterator):
                loss_val, train_acc, _ = model(data['images'], data['labels'],
                                               (step - 1) * train_ds_len + b_id)
                loss_val_step.append(loss_val)
                train_acc_step.append(train_acc)
                progbar_train.update(
                    b_id + 1, values=[('loss', loss_val), ('accuracy', train_acc)])

            loss_val_avg.append(sum(loss_val_step) / len(loss_val_step))
            train_acc_avg.append(sum(train_acc_step) / len(train_acc_step))

            loss_val = loss_val_avg[-1]
            train_acc = train_acc_avg[-1]

            if step % cfg.train_sum_every == 0:
                summary_writer.flush()

                fd["loss"].write("{:d},{:.4f}\n".format(step, loss_val))
                fd["loss"].flush()
                fd["train_acc"].write("{:d},{:.4f}\n".format(step, train_acc))
                fd["train_acc"].flush()

            if step % cfg.val_sum_every == 0:
                print("evaluating, it will take a while...")
                probs = []
                targets = []
                total_acc = 0
                n = 0
                for b_id, data in enumerate(validation_iterator):
                    prob, _, val_acc = model.eval(data['images'], data['labels'])
                    probs.append(prob)
                    targets.append(data['labels'])
                    total_acc += val_acc
                    n += 1
                    progbar_val.update(b_id + 1, values=[('accuracy', val_acc)])

                probs = np.concatenate(probs, axis=0)
                targets = np.concatenate(targets, axis=0).reshape((-1, 1))
                avg_acc = total_acc / n
                path = os.path.join(os.path.join(cfg.results_dir, "activations"))
                plot_activation(np.hstack((probs, targets)), step=step, save_to=path)
                fd["val_acc"].write("{:d},{:.4f}\n".format(step, avg_acc))
                fd["val_acc"].flush()

            if step % cfg.save_ckpt_every == 0:
                model.save_weights(checkpoint_path.format(epoch=0))

            duration = time.time() - start_time
            log_str = ' step: {:d}, loss: {:.3f}, accuracy: {:.3f}, time: {:.3f} sec/step'.format(step, loss_val, train_acc, duration)
            print(log_str)


def evaluate(model, data_loader):
    # Setting up model
    test_iterator = data_loader(cfg.batch_size, mode="test")

    # Create files to save evaluating results
    fd = save_to(is_training=False)

    # Compute the cardinality of the dataset
    ds_len = 0
    for _ in test_iterator:
        ds_len = ds_len + 1

    # Initialize progress bar
    progbar = tf.keras.utils.Progbar(ds_len)

    # Load the model
    model.load_weights(tf.train.latest_checkpoint(cfg.logdir))
    print('Model restored!')

    # Test
    probs = []
    targets = []
    total_acc = 0
    n = 0

    for b_id, data in enumerate(test_iterator):
        prob, label, test_acc = model.eval(data['images'], data['labels'])
        probs.append(prob)
        targets.append(label)
        total_acc += test_acc
        n += 1
        progbar.update(b_id + 1, values=[('accuracy', test_acc)])

    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0).reshape((-1, 1))
    avg_acc = total_acc / n
    out_path = os.path.join(cfg.results_dir, 'prob_test.txt')
    np.savetxt(out_path, np.hstack((probs, targets)), fmt='%1.2f')
    print('Classification probability for each category has been saved to ' + out_path)
    fd["test_acc"].write(str(avg_acc))
    fd["test_acc"].close()
    out_path = os.path.join(cfg.results_dir, 'test_accuracy.txt')
    print('Test accuracy has been saved to ' + out_path)


def main():
    model_list = ['baseline', 'vectorCapsNet', 'matrixCapsNet', 'convCapsNet']

    # Deciding which model to use
    if cfg.model == 'baseline':
        model = import_module(cfg.model).Model
    elif cfg.model in model_list:
        model = import_module(cfg.model).CapsNet
    else:
        raise ValueError('Unsupported model, please check the name of model:', cfg.model)

    # Deciding which dataset to use
    if cfg.dataset == 'mnist' or cfg.dataset == 'fashion_mnist':
        height = 28
        width = 28
        channels = 1
        num_label = 10
    elif cfg.dataset == 'smallNORB':
        num_label = 5
        height = 32
        width = 32
        channels = 1

    # Initializing model and data loader
    net = model(height=height, width=width,
                channels=channels, num_label=num_label)
    dataset = "capslayer.data.datasets." + cfg.dataset
    data_loader = import_module(dataset).DataLoader(path=cfg.data_dir,
                                                    splitting=cfg.splitting,
                                                    num_works=cfg.num_works)

    # Deciding to train or evaluate model
    if cfg.is_training:
        train(net, data_loader)
    else:
        evaluate(net, data_loader)


if __name__ == "__main__":
    main()
