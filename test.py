import time
import scipy.signal
import numpy as np
import random

import keras
from keras.preprocessing.text import Tokenizer

from matplotlib import pyplot as plt

import tensorflow as tf

def load_data():
    fp = open("simpler_signal", "rb")
    log_data = fp.read()
    fp.close()

    # don't die on invalid utf characters, just skip them
    return log_data.decode("utf-8", "ignore")


def dothething(i, k):
    i = tf.constant(i, dtype=tf.float32, name="i")
    k = tf.constant(k, dtype=tf.float32, name="k")

    dc_offset = tf.math.reduce_mean(i)

    i = tf.subtract(i, dc_offset)
    k = tf.subtract(k, dc_offset)

    data = tf.reshape(i, [1, int(i.shape[0]), 1], name="data")
    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name="kernel")

    # not actually the activated res, why do I suck
    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))
    res_max = tf.math.reduce_max(res)
    res_min = tf.math.reduce_min(res)

    activated_res = tf.keras.activations.selu(res)

    with tf.Session() as sess:
        result, activated_res = sess.run([res, activated_res])
        res_max, res_min, dc_offset = sess.run([res_max, res_min, dc_offset])

        plt.plot(result)
        plt.ylabel("some numbers")
        plt.savefig(f"graphs/{time.time()}.png")


def hipass_filter(raw_vector):
    kernel = [0.25, 0.25, 0.25, 0.25]

    raw_tensor = tf.constant(raw_vector, dtype=tf.float32, name="i")
    filter_tensor = tf.constant(kernel, dtype=tf.float32, name="k")

    data = tf.reshape(raw_tensor, [1, int(raw_tensor.shape[0]), 1], name="data")
    kernel = tf.reshape(filter_tensor, [int(filter_tensor.shape[0]), 1, 1], name="kernel")

    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))

    with tf.Session() as sess:
        result = sess.run(res)

    return result

def numpy_dothething():
    dc_offset = np.mean(vectorized_data)
    vectorized_data = vectorized_data - dc_offset
    filter_signal = filter_signal - dc_offset

    vectorized_data = [0] + vectorized_data + [0]
    filter_signal = [0] + filter_signal + [0]

    convolved = scipy.signal.convolve(vectorized_data, filter_signal, mode="same")

    plt.plot(convolved)
    plt.ylabel("some numbers")
    plt.savefig(f"graphs/scipy-{time.time()}.png")


if __name__ == "__main__":
    raw_data = load_data()
    tokenizer = Tokenizer(filters=None, char_level=True)
    tokenizer.fit_on_texts([raw_data])

    vectorized_data, filter_signal = tokenizer.texts_to_sequences([raw_data, "error"])

    centered_data = hipass_filter(vectorized_data)
    raw_data = dothething(centered_data, filter_signal)
