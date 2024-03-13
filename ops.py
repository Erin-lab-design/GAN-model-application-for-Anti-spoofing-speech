import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer

def conv1d(x, kwidth=5, num_kernels=1, init=None, uniform=False, bias_init=None, name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    w_init = init
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, in_channels, num_kernels], initializer=w_init)
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.get_variable('b', [num_kernels], initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv

def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)

def prelu(x, name='prelu', ref=False):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', in_shape[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * 0.5
        if ref:
            return pos + neg, alpha
        else:
            return pos + neg
        
def conv2d(input_, output_dim, k_h, k_w, stddev=0.05, name="conv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], 
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
        if with_w:
            return conv, w
        else:
            return conv