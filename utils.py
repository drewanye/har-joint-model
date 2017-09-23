
from __future__ import division
__author__ = 'zhenanye'

import tensorflow as tf
import inspect
import numpy as np

# def batch_norm(x, name, is_training):
#     with tf.variable_scope(name):
#         params_shape = [x.get_shape()[-1]]
#
#         beta = tf.get_variable(
#             'beta', params_shape, tf.float32,
#             initializer=tf.constant_initializer(0.0, tf.float32))
#         gamma = tf.get_variable(
#             'gamma', params_shape, tf.float32,
#             initializer=tf.constant_initializer(1.0, tf.float32))
#
#         if is_training == True:
#             mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
#
#             moving_mean = tf.get_variable(
#                 'moving_mean', params_shape, tf.float32,
#                 initializer=tf.constant_initializer(0.0, tf.float32),
#                 trainable=False)
#             moving_variance = tf.get_variable(
#                 'moving_variance', params_shape, tf.float32,
#                 initializer=tf.constant_initializer(1.0, tf.float32),
#                 trainable=False)
#
#             self._extra_train_ops.append(moving_averages.assign_moving_average(
#                 moving_mean, mean, 0.9))
#             self._extra_train_ops.append(moving_averages.assign_moving_average(
#                 moving_variance, variance, 0.9))
#         else:
#             mean = tf.get_variable(
#                 'moving_mean', params_shape, tf.float32,
#                 initializer=tf.constant_initializer(0.0, tf.float32),
#                 trainable=False)
#             variance = tf.get_variable(
#                 'moving_variance', params_shape, tf.float32,
#                 initializer=tf.constant_initializer(1.0, tf.float32),
#                 trainable=False)
#             tf.summary.histogram(mean.op.name, mean)
#             tf.summary.histogram(variance.op.name, variance)
#         # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
#         y = tf.nn.batch_normalization(
#             x, mean, variance, beta, gamma, 0.001)
#         y.set_shape(x.get_shape())
#         return y

# def batch_norm(x, is_training, name='batch_norm'):
#     with tf.variable_scope(name):
#         x_shape = x.get_shape()
#         batch_mean, batch_var = tf.nn.moments(x, axes=range(0, len(x_shape)-1), name='momentss')
#         print("******batch_mean:{}***".format(batch_mean.name))
#         params_shape = [x_shape[-1]]
#         beta = tf.get_variable(
#             'beta', params_shape, tf.float32,
#             initializer=tf.constant_initializer(0.0, tf.float32))
#         print("****beta:{}*****".format(beta.name))
#         gamma = tf.get_variable(
#             'gamma', params_shape, tf.float32,
#             initializer=tf.constant_initializer(1.0, tf.float32))
#
#         ema = tf.train.ExponentialMovingAverage(decay=0.9)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(is_training, mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#
#         y = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)
#         y.set_shape(x.get_shape())
#         return y

def batch_norm(x, is_training, name='batch_norm'):
    return tf.layers.batch_normalization(x, momentum=0.9, center=True, scale=True, training=is_training)


def input_batch_norm(x, name='input_norm'):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))

        y = tf.nn.batch_normalization(x, mean, var, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y

def weight_variable(shape):
    w = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(stddev=0.1))
    # w = tf.Variable(tf.random_normal(shape))
    return w

def bias_variable(shape):
    # return tf.Variable(tf.random_normal(shape))
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))

def conv2d(x, W, padding):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

def max_pool(x, ksize_height, ksize_width, s_height, s_width, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, ksize_height, ksize_width, 1], strides=[1, s_height, s_width, 1], padding=padding)

def conv(input, kernel_shape, bias_shape, padding='SAME'):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)
    h_conv = conv2d(input, weights, padding)
    return h_conv + bias
    # return tf.nn.relu(h_conv+bias)

def activation(x):
    return tf.nn.relu(x)
    # return selu(x)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0, x, alpha*tf.nn.elu(x))

def full_conn(input, kernel_shape, bias_shape):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)
    # if norm:
    #     input = batch_norm(input, is_training=is_training)
    return tf.matmul(input, weights) + bias
    # return tf.nn.relu(tf.matmul(input, weights) + bias)

def scores(input, kernel_shape, bias_shape):
    weights = weight_variable(kernel_shape)
    bias = bias_variable(bias_shape)

    return tf.matmul(input, weights) + bias

def lstm_cell(size):
    # With the latest TensorFlow source code (as of Mar 27, 2017),
    # the BasicLSTMCell will need a reuse parameter which is unfortunately not
    # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
    # an argument check here:
    if 'reuse' in inspect.getargspec(
            tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
    else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)

def normalize(X, axis=0):
    X -= np.mean(X, axis=axis)
    X /= np.std(X, axis=axis)

    return X

def one_hot(y_, n_values):
    #y_ = y_.reshape(len(y_))
    # n_values = int(np.max(y_))+1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def f1_score(cf_matrix, epsilon=1e-5):
    class_num = cf_matrix.shape[0]
    size = np.sum(cf_matrix)
    weights = []
    precision = []
    recall = []
    for i in range(class_num):
        tp = cf_matrix[i, i]
        tp_fp = np.sum(cf_matrix[:, i])
        tp_fn = np.sum(cf_matrix[i, :])
        weights.append(tp_fn/size)
        precision.append(tp/(tp_fp+epsilon))
        recall.append(tp/(tp_fn+epsilon))
    weights = np.asarray(weights, dtype=np.float32)
    precision = np.asarray(precision, dtype=np.float32)
    recall = np.asarray(recall, dtype=np.float32)
    f1 = 2 * (precision*recall) / (precision+recall+epsilon)
    weighted_f1 = weights * f1
    return np.sum(weighted_f1)




# from http://www.johnvinyard.com/blog/?p=268

# import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


