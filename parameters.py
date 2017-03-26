import tensorflow as tf
from operator import mul
import functools

std_filter_size = 3
std_n_filters = 32
depth=std_n_filters

def weights(filter_size=std_filter_size, depth=depth, n_filters=std_n_filters):
    weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, depth, n_filters], stddev=0.1))

    return weights

def biases(n_filters=std_n_filters):
    biases = tf.Variable(tf.constant(0.1, shape=[n_filters]), name="bias")

    return biases

def fc_weights(data, output):
    get_size = functools.reduce(mul, data.get_shape()[1:])

    weights = tf.Variable(tf.truncated_normal([int(get_size), output]))

    return weights
