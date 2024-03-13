from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from ops import *
import numpy as np

def discriminator(self, wave_in, reuse=False):
    """
    波形输入判别器,输出连续值分数
    """
    in_dims = wave_in.get_shape().as_list()
    hi = wave_in
    if len(in_dims) == 2:
        hi = tf.expand_dims(wave_in, -1)
    elif len(in_dims) < 2 or len(in_dims) > 3:
        raise ValueError('Discriminator input must be 2-D or 3-D')

    batch_size = int(wave_in.get_shape()[0])

    with tf.variable_scope('d_model') as scope:
        if reuse:
            scope.reuse_variables()
        def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation, pooling=2):
            with tf.variable_scope('d_block_{}'.format(block_idx)):
                bias_init = None
                if self.bias_D_conv:
                    bias_init = tf.constant_initializer(0.)
                downconv_init = tf.truncated_normal_initializer(stddev=0.02)
                hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                init=downconv_init, bias_init=bias_init)
                if bnorm:
                    hi_a = self.vbn(hi_a, 'd_vbn_{}'.format(block_idx))
                if activation == 'leakyrelu':
                    hi = leakyrelu(hi_a)
                elif activation == 'relu':
                    hi = tf.nn.relu(hi_a)
                else:
                    raise ValueError('Unrecognized activation {} in D'.format(activation))
                return hi

        for block_idx, fmaps in enumerate(self.d_num_fmaps):
            hi = disc_block(block_idx, hi, 31, self.d_num_fmaps[block_idx], True, 'leakyrelu')

        hi_f = flatten(hi)
        d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                             init=tf.truncated_normal_initializer(stddev=0.02),
                             name='logits_conv')
        d_logit_out = tf.squeeze(d_logit_out, -1)
        d_score = tf.nn.sigmoid(d_logit_out)

        return d_score

# 计算二分类交叉熵损失
def discriminator_loss(self, y_true, y_pred):
    y_true = tf.placeholder(tf.float32, shape=(None, 1))
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(losses)
    return loss