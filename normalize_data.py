import time

import graphyte
import petname

graphyte.init("graphite.etsycorp.com", prefix="akarpinski")

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy
import os

from bits_n_bobs import bigtext, plot_vector

run_id = petname.generate()
print(run_id)

def save_prepared_tensor(tensor):
    save_to_path = f"runs/{run_id}/normalized-{time.time()}.npy"

    run_dir = os.path.dirname(save_to_path)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    numpy.save(save_to_path, tensor)

def prepare_vector(raw_vector):
    raw_tensor = tf.convert_to_tensor(value=raw_vector)
    prepared_tensor = tf.Variable(0)

    variance_epsilon = tf.constant(0.1)

    mean, variance = tf.nn.moments(x=raw_tensor, axes=[0])

    prepared_tensor = tf.nn.batch_normalization(
        raw_tensor,
        mean,
        variance,
        offset=None,
        scale=None,
        variance_epsilon=variance_epsilon,
    )

    return prepared_tensor


if __name__ == "__main__":
    hyper_before = time.time()  # a better name might be start_time

    # LOAD
    before = time.time()
    with open("info.log.large", "rb") as fp:
        numpy_arr_of_letters = numpy.array(list(fp.read()), dtype=numpy.float32)

    # plot_vector(numpy_arr_of_letters)
    graphyte.send(f"tf.{run_id}.prep.load", time.time() - before)

    before = time.time()
    dataset_size = len(numpy_arr_of_letters)
    graphyte.send(f"tf.{run_id}.prep.count_len", time.time() - before)
    graphyte.send(f"tf.{run_id}.prep.dataset_size", dataset_size)

    chunked_dataset = numpy.array_split(numpy_arr_of_letters, 3)
    results = []
    for chunk in chunked_dataset:
        print("processing new chunk")
        before = time.time()
        prepared_tensor = prepare_vector(chunk)
        graphyte.send(f"tf.{run_id}.prep.normalize", time.time() - before)

        before = time.time()
        save_prepared_tensor(prepared_tensor)
        graphyte.send(f"tf.{run_id}.prep.write", time.time() - before)


    graphyte.send(f"tf.{run_id}.prep.total", time.time() - hyper_before)

    print(bigtext(run_id.upper()))
    print(run_id)
