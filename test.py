import time
import scipy.signal
import numpy as np
import random

import keras
from keras.preprocessing.text import Tokenizer

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_transform as tft


def load_data():
    fp = open("info.log.smaller", "rb")
    log_data = fp.read()
    fp.close()

    # don't die on invalid utf characters, just skip them
    return log_data.decode("utf-8", "ignore")


def dothething(i, k):
    data = tf.reshape(i, [1, int(i.shape[0]), 1], name="data")
    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name="kernel")

    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))
    res_max = tf.math.reduce_max(res)
    res_min = tf.math.reduce_min(res)

    activated_res = tf.keras.activations.selu(res)

    with tf.Session() as sess:
        result, activated_res = sess.run([res, activated_res])
        res_max, res_min = sess.run([res_max, res_min])
        i = sess.run(i)
        print(res_max, res_min)

        plt.plot(result)
        plt.plot(i)
        plt.ylabel("some numbers")
        plt.savefig(f"graphs/{time.time()}.png")


def remove_dc_offset(raw_vector):
    raw_tensor = tf.constant(raw_vector, dtype=tf.float16)
    signal_mean = tf.reduce_mean(raw_tensor)
    centered_signal = raw_tensor - signal_mean

    with tf.Session() as sess:
        return sess.run(centered_signal)


def divide_variance(raw_vector):
    pass


def prepare_vector(raw_vector):
    raw_tensor = tf.constant(raw_vector, dtype=tf.float32)
    scaled_tensor = tft.scale_to_z_score(raw_tensor)

    return scaled_tensor


if __name__ == "__main__":
    raw_data = load_data()
    tokenizer = Tokenizer(filters=None, char_level=True)
    tokenizer.fit_on_texts([raw_data])

    filter_string = 'Eurl2IgZokV-pfFZkfOAZw4HCA67'

    vectorized_data, filter_signal = tokenizer.texts_to_sequences(
        [raw_data, filter_string]
    )

    prepared_tensor = prepare_vector(vectorized_data)
    prepared_filter = prepare_vector(filter_signal)

    raw_data = dothething(prepared_tensor, prepared_filter)
