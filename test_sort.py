import tensorflow as tf


def softsort(b1, b2, c, tau=16, pow=1):
    s = tf.einsum('nc,kc->nk', b1, b2)
    s = tf.expand_dims(s, axis=-1)
    s_sorted = tf.sort(s, direction='DESCENDING', axis=1)
    pairwise_distances = -tf.pow(tf.abs(tf.transpose(s, perm=[0, 2, 1]) - s_sorted), pow)
    P_hat = tf.nn.softmax(pairwise_distances / tau, -1)

    z = tf.nn.l2_normalize(tf.matmul(P_hat, c), axis=-1)

    return P_hat



a = tf.constant([[0, 1, 1], [0, 1, 0]])


b = tf.ones([3,3,5])
c = tf.ones([3,5])

print(tf.einsum('nkd,kd->nk', b, c))