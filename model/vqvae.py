from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
import tensorflow as tf


class ToyModel(tf.keras.Model):
    def __init__(self, feat_size, layer=4, k=10, **kwargs):
        super().__init__(**kwargs)
        # couplings += [CouplingLayer(feat_size, feat_size * 2, tf.identity)]

        self.feat_size = feat_size
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(int(feat_size / 2), activation='relu'),
            tf.keras.layers.Dense(int(feat_size / 28))
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(int(feat_size / 2), activation='relu'),
            tf.keras.layers.Dense(feat_size)
        ])
        self.k = k
        initializer = tf.keras.initializers.GlorotUniform()
        self.prior = tf.Variable(trainable=True, initial_value=initializer(shape=[k, int(feat_size / 28)]))

    def call(self, inputs, training=True, mask=None, step=-1):
        _z = self.encoder(inputs, training=training)
        d = self.pairwise_dist(_z)
        target_ind = tf.argmin(d, axis=1)
        target = tf.gather(self.prior, target_ind)
        z = tf.stop_gradient(target - _z) + _z

        _x = self.decoder(z, training=training)
        if training:
            loss_1 = tf.reduce_mean(tf.reduce_sum(tf.pow(_z - tf.stop_gradient(target), 2), axis=1) / 2)
            loss_2 = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.stop_gradient(_z) - target, 2), axis=1) / 2)

            loss_rec = tf.reduce_mean(tf.reduce_sum(tf.pow(inputs - _x, 2), axis=1))
            loss_vq = loss_1 + loss_2 * .25
            loss = loss_vq + loss_rec
            self.add_loss(loss)
            if step >= 0:
                _g = self.decoder(self.prior)
                img = self.show_mnist(_g)
                __x = tf.stop_gradient(_x)
                tf.summary.histogram('assign', target_ind, step=step)
                tf.summary.scalar('loss/loss_all', loss, step=step)
                tf.summary.scalar('rec', loss_rec, step=step)
                tf.summary.image('img', img, step=step, max_outputs=1)
        return _z


    def pairwise_dist(self, a):
        na = tf.reduce_sum(tf.square(a), 1)
        nb = tf.reduce_sum(tf.square(self.prior), 1)

        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        rslt = tf.maximum(na - 2 * tf.matmul(a, self.prior, False, True) + nb, 0.0)
        return rslt

    def show_mnist(self, feat):
        size = int(np.sqrt(self.feat_size))
        img = tf.reshape(feat, [-1, size, size]).numpy()
        rslt = np.zeros([int(self.k / 3 + 1) * size, 3 * size])
        for i in range(self.k):
            _r = int(i / 3) * size
            _c = i % 3 * size
            rslt[_r:(_r + size), _c:(_c + size)] = img[i, :, :]

        return rslt[np.newaxis, :, :, np.newaxis]

