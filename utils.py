__author__ = 'zhenanye'

import tensorflow as tf

def weight_variable(shape):
    w = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    return w

def bias_variable(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, ksize_height, ksize_width, s_height, s_width):
  return tf.nn.max_pool(x, ksize=[1, ksize_height, ksize_width, 1], strides=[1, s_height, s_width, 1], padding='SAME')

def conv_relu(input, kernel_shape, bias_shape): #  norm=False, is_training=False):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)
    h_conv = conv2d(input, weights)
    # if norm:
    #     h_conv = tf.layers.batch_normalization(h_conv, momentum=0.4, training=is_training)
    return tf.nn.relu(h_conv+bias)

def full_conn_relu(input, kernel_shape, bias_shape, norm=False, is_training=False):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)
    if norm:
        input = tf.layers.batch_normalization(input, momentum=0.4, training=is_training)
    return tf.nn.relu(tf.matmul(input, weights) + bias)

def scores(input, kernel_shape, bias_shape):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)

    return tf.matmul(input, weights) + bias
