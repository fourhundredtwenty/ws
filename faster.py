import time
import scipy.signal
import numpy as np
import random

from matplotlib import pyplot as plt

import graphyte
graphyte.init('graphitegcp.etsycorp.com', prefix='akarpinski')

import tensorflow as tf
import tensorflow_transform as tft


def load_data(path):
    dataset = tf.data.TextLineDataset(path)
    return dataset

    # with open(path, 'rb') as fp:
    #     return list(fp.read())

def dothething(i, k):
    data = tf.reshape(i, [1, int(i.shape[0]), 1], name="data")
    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name="kernel")

    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))
    res_max = tf.math.reduce_max(res)
    res_min = tf.math.reduce_min(res)

    with tf.Session() as sess:
        result = sess.run(res)

        # result, activated_res = sess.run([res, activated_res])
        # res_max, res_min = sess.run([res_max, res_min])
        # i = sess.run(i)
        # print(res_max, res_min)

        # plt.plot(result)
        # plt.plot(i)
        # plt.ylabel("some numbers")
        # plt.savefig(f"graphs/{time.time()}.png")

def prepare_vector(raw_tensor):
    numeric_tensor = tf.strings.to_number(raw_tensor)
    # mean, variance = tf.nn.moments(numeric_tensor, axes=[0])
    # prepared_tensor = tf.nn.batch_normalization(numeric_tensor, mean, variance, offset=None, scale=None, variance_epsilon=.01)
    return prepared_tensor

    # raw_tensor = tf.constant(raw_vector, dtype=tf.float32)
    # scaled_tensor = tft.scale_to_z_score(raw_tensor)

    # return scaled_tensor


if __name__ == "__main__":
    before = time.time()
    vectorized_data = load_data("info.log.smaller")
    load_time = time.time()-before
    graphyte.send('tf.load', load_time)

    filter_string = b'Eurl2IgZokV-pfFZkfOAZw4HCA67'
    filter_signal = list(filter_string)

    before = time.time()
    prepared_dataset = vectorized_data.map(prepare_vector)
    exit

    prepared_filter = prepare_vector(filter_signal)
    prep_time = time.time()-before
    graphyte.send('tf.prep', prep_time)

    before = time.time()
    processed_data = dothething(prepared_tensor, prepared_filter)
    search_time = time.time()-before
    graphyte.send('tf.search', search_time)
