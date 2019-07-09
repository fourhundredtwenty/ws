import time
import numpy
import tensorflow as tf


def load_data():
    with open("/usr/share/dict/words", "r") as fp:
        print(list(fp.read())[200:300])
    with open("/usr/share/dict/words", "rb") as fp:
        numpy_arr_of_letters = numpy.array(list(fp.read()), dtype=numpy.float32)
    return numpy_arr_of_letters


if __name__ == "__main__":

    raw_data = load_data()
    print(raw_data[200:300])
