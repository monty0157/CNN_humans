import parameters as params
import tensorflow as tf
from operator import mul
import functools

conv_stride = [1,1,1,1]
conv_padding = 'SAME'
pool_stride = [1,2,2,1]
pool_padding = 'SAME'
pool_ksize = [1,2,2,1]


def conv_layer(data, weights, biases, strides=conv_stride, padding=conv_padding):
    convolution = tf.nn.relu(tf.nn.conv2d(data, weights, strides=strides, padding=padding) + biases)
    return convolution

def pool_layer(data, ksize=pool_ksize, strides=pool_stride, padding=pool_padding):
    pool = tf.nn.max_pool(data, ksize=ksize, strides=strides, padding=padding)

    return pool

def full_layer(data, weights, biases, dropout):
    if len(data.get_shape()) is 4:
        get_size = functools.reduce(mul, data.get_shape()[1:])
        data = tf.reshape(data, [-1, int(get_size)])

    fc = tf.nn.relu(tf.matmul(data, weights) + biases)

    fc = tf.nn.dropout(fc, dropout)

    return fc

def output_layer(data, weights, biases):
    output = tf.matmul(data, weights) + biases

    return output
