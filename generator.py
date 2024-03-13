from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *

class Generator(object):
    def __init__(self, segan):
        self.segan = segan

    def __call__(self, input_audio, is_ref, spk=None):
        """ Build the graph propagating (input_audio) --> x
        On first pass will make variables.
        """
        segan = self.segan

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True

        print('*** Building Generator ***')
        in_dims = input_audio.get_shape().as_list()
        h_i = input_audio
        if len(in_dims) == 2:
            h_i = tf.expand_dims(input_audio, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 3
        skips = []
        for block_idx, dilation in enumerate(segan.g_dilated_blocks):
            name = 'g_residual_block_{}'.format(block_idx)
            if block_idx >= len(segan.g_dilated_blocks) - 1:
                skip_out = False
            if skip_out:
                res_i, skip_i = residual_block(
                    h_i,
                    dilation,
                    kwidth,
                    num_kernels=32,
                    bias_init=None,
                    stddev=0.02,
                    do_skip=True,
                    name=name)
            else:
                res_i = residual_block(
                    h_i,
                    dilation,
                    kwidth,
                    num_kernels=32,
                    bias_init=None,
                    stddev=0.02,
                    do_skip=False,
                    name=name)
            h_i = res_i
            if skip_out:
                skips.append(skip_i)
            else:
                skips.append(res_i)

        print('Amount of skip connections: ', len(skips))
        with tf.variable_scope('g_wave_pooling'):
            skip_T = tf.stack(skips, axis=0)
            skips_sum = tf.reduce_sum(skip_T, axis=0)
            skips_sum = leakyrelu(skips_sum)
            wave_a = conv1d(
                skips_sum,
                kwidth=1,
                num_kernels=1,
                init=tf.truncated_normal_initializer(stddev=0.02))
            wave = tf.tanh(wave_a)

        print('Last residual wave shape: ', res_i.get_shape())
        print('*************************')
        segan.generator_built = True
        return wave