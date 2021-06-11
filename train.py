from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from meta import ROOT_PATH

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_mnist(batch_size):
    def _map(x):
        x['feat'] = tf.cast(tf.reshape(x['image'], [batch_size, -1]), dtype=tf.float32) / 255.
        return x

    data: tf.data.Dataset = tfds.load('mnist', split='train', data_dir=os.path.join(ROOT_PATH, 'data'))

    data = data.shuffle(30000).batch(batch_size).map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return data


if __name__ == '__main__':
    ds = get_mnist(12)
    for d in ds:
        print(d)
        break
