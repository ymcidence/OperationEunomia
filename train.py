from __future__ import absolute_import, print_function, division, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from meta import ROOT_PATH
from time import gmtime, strftime
from model.toy_model import ToyModel as Model

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_mnist(batch_size):
    def _map(x):
        x['feat'] = tf.cast(tf.reshape(x['image'], [-1]), dtype=tf.float32) / 255.
        return x

    data: tf.data.Dataset = tfds.load('mnist', split='train', data_dir=os.path.join(ROOT_PATH, 'data'))

    data = data.shuffle(30000).map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return data


def step_train(data: dict, model: Model, opt: tf.keras.optimizers.Optimizer, step):
    # label = data['label']label
    _step = -1 if step % 100 > 0 else step

    if step == 0:
        _ = model(data['feat'], step=-1, training=False)
        model.update_initial()

    with tf.GradientTape() as tape:
        _ = model(data['feat'], step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.numpy()


def epoch_train(data: tf.data.Dataset, model: Model, opt: tf.keras.optimizers.Optimizer, step_count=0):
    this_step = step_count + 0

    for i, d in enumerate(data):
        this_step += 1

        loss = step_train(d, model, opt, this_step)

        if this_step % 50 == 0:
            print('iter {}, loss {}'.format(this_step, loss))

    return this_step


def main(task_name='mnist'):
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())

    result_path = os.path.join(ROOT_PATH, 'result', 'mnist')
    save_path = os.path.join(result_path, 'model', task_name + '_' + time_string)
    summary_path = os.path.join(result_path, 'log', task_name + '_' + time_string)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Model(feat_size=28 * 28, k=20)
    data = get_mnist(256)
    opt = tf.keras.optimizers.Adam(1e-4)
    writer = tf.summary.create_file_writer(summary_path)
    # checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)
    starter = 0
    with writer.as_default():
        for i in range(1000):
            print('This is epoch {}'.format(i))
            starter = epoch_train(data, model, opt, starter)


if __name__ == '__main__':
    main()
