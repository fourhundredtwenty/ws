import time

from matplotlib import pyplot as plt

import graphyte

graphyte.init("graphitegcp.etsycorp.com", prefix="akarpinski")

import tensorflow as tf
import tensorflow_transform as tft

tf.enable_eager_execution()


def dothething(i, k):
    data = tf.reshape(i, [1, int(i.shape[0]), 1], name="data")
    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name="kernel")

    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))
    res_max = tf.math.reduce_max(res)
    res_min = tf.math.reduce_min(res)

    with tf.Session() as sess:
        result = sess.run(res)

def prepare_vector(raw_tensor):
    numeric_tensor = tf.strings.to_number(raw_tensor)
    mean, variance = tf.nn.moments(numeric_tensor, axes=[0])
    prepared_tensor = tf.nn.batch_normalization(
        numeric_tensor, mean, variance, offset=None, scale=None, variance_epsilon=0.01
    )
    return prepared_tensor

if __name__ == "__main__":
    dataset = tf.data.TextLineDataset(["info.log.smaller"])
    prepared_dataset = dataset.batch(1000).map(prepare_vector)

    for datum in prepared_dataset:
        print(datum)
