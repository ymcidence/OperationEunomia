from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


class CouplingLayer(tf.keras.layers.Layer):
    def __init__(self, feat_size, hidden=128, activation=tf.nn.leaky_relu, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = feat_size
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(hidden, activation=activation))
        self.net.add(tf.keras.layers.Dense(feat_size, activation=activation))

    def call(self, x, training=True, forward=True, **kwargs):
        if forward:
            x_1 = x[:, :self.input_dim // 2]
            x_2 = x[:, self.input_dim // 2:]

            h = self.net(x_1)
            shift = h[:, 0::2]
            scale = tf.nn.sigmoid(h[:, 1::2] + 2.) + 1e-8

            x_2 = x_2 + shift
            x_2 = x_2 * scale

            jac = tf.reduce_sum(tf.math.log(scale), 1)

            return tf.concat([x_1, x_2], 1), jac

        else:
            x_1 = x[:, :self.input_dim // 2]
            x_2 = x[:, self.input_dim // 2:]

            h = self.net(x_1)
            shift = h[:, 0::2]
            scale = tf.nn.sigmoid(h[:, 1::2] + 2.) + 1e-8
            x_2 /= scale
            x_2 -= shift

            jac = tf.reduce_sum(tf.math.log(scale), 1) * -1
            return tf.concat([x_1, x_2], 1), jac
