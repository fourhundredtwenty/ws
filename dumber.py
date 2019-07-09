import time

from matplotlib import pyplot as plt

import graphyte

graphyte.init("graphite.etsycorp.com", prefix="akarpinski")

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.client import timeline
import numpy

def dothething(i, k, datums):
    data = tf.reshape(i, [1, int(i.shape[0]), 1])
    kernel = tf.reshape(k, [int(k.shape[0]), 1, 1])

    res = tf.squeeze(tf.nn.conv1d(data, kernel, 1, "SAME"))

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    run_metadata = tf.RunMetadata()


    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)
        res = sess.run(res, feed_dict={'data:0': datums}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        #res = sess.run(kernel, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        #res = sess.run(res)# , options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        with open(f"./timelines/timeline-{time.time()}.ctf.json", 'w') as trace_file:
               trace_file.write(trace.generate_chrome_trace_format())

    return res
    
def prepare_vector(raw_tensor):
    variance_epsilon = tf.constant(0.1)
    mean, variance = tf.nn.moments(raw_tensor, axes=[0])
    prepared_tensor = tf.nn.batch_normalization(
        raw_tensor, mean, variance, offset=None, scale=None, variance_epsilon=variance_epsilon
    )
    return prepared_tensor

if __name__ == "__main__":
    hyper_before = time.time() # a better name might be start_time
    search_string = b"this is fucking wild right now, you know"
    kernel = numpy.array(list(search_string), dtype=numpy.float32)

    before = time.time()

    with open("info.log.medium", "rb") as fp:
        numpy_arr_of_letters = numpy.array(list(fp.read()), dtype=numpy.float32)
    full_dataset_len = len(numpy_arr_of_letters)

    kernel = tf.get_variable("kernel", initializer=kernel)
    kernel = prepare_vector(kernel)

    #dataset = tf.get_variable("data", initializer=numpy_arr_of_letters)


    number_of_batches = 2
    chunked_dataset = numpy.array_split(numpy_arr_of_letters, number_of_batches)
    dataset = tf.placeholder(tf.float32, name="data", shape=[len(chunked_dataset[0])])
    for chunk in chunked_dataset:
        before = time.time()
        dataset = prepare_vector(dataset) 
        graphyte.send('tf.prep', time.time()-before)

        before = time.time()
        result = dothething(dataset, kernel, chunk)
        graphyte.send('tf.convolve', time.time()-before)

    graphyte.send('tf.full_time', time.time()-hyper_before)

