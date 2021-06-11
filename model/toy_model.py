from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np
import tensorflow as tf

from layer.coupling import CouplingLayer


class ToyModel(tf.keras.Model):
    def __init__(self, feat_size, layer=4, k=10, **kwargs):
        super().__init__(**kwargs)
        couplings = [CouplingLayer(feat_size, feat_size * 2) for _ in range(layer - 1)]
        couplings += [CouplingLayer(feat_size, feat_size * 2, tf.identity)]
        self.inn = couplings
        self.feat_size = feat_size
        self.k = k
        initializer = tf.keras.initializers.GlorotUniform()
        self.prior = tf.Variable([k, feat_size], trainable=True, initial_value=initializer(shape=[k, feat_size]))

    def call(self, inputs, training=True, mask=None, step=-1):
        batch_size = tf.shape(inputs)[0]
        _x = inputs
        _jac = tf.zeros([batch_size])

        for l in self.inn:
            _x, _j = l(_x, training=True, forward=True)  # [N D], [N]
            _jac += _j  # [N]

        _g = tf.stop_gradient(self.prior)
        for i in range(self.inn.__len__()):
            _g, _ = self.inn[-1 * i](_g, forward=False)

        if training:
            d = self.pairwise_dist(_x)

            target_ind = tf.argmin(d, axis=1)
            target = tf.gather(self.prior, target_ind)

            loss_1 = tf.reduce_mean(tf.reduce_sum(tf.pow(_x - tf.stop_gradient(target), 2), axis=1) / 2)
            loss_2 = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.stop_gradient(_x) - target, 2), axis=1) / 2)
            loss_vq = loss_1 + loss_2 * .25

            loss = loss_vq - tf.reduce_mean(_jac)
            self.add_loss(loss)

            if step >= 0:
                img = self.show_mnist(_g)
                tf.summary.histogram('assign', target_ind, step=step)
                tf.summary.scalar('loss/loss_all', loss, step=step)
                tf.summary.scalar('loss/jac', tf.reduce_mean(_jac), step=step)
                tf.summary.image('img', img, step=step, max_outputs=1)

        return _x

    def pairwise_dist(self, a):
        na = tf.reduce_sum(tf.square(a), 1)
        nb = tf.reduce_sum(tf.square(self.prior), 1)

        # na as a row and nb as a co"lumn vectors
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

        return rslt
