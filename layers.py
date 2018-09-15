# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import numpy as np
def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
	x_shape = tf.shape(x)
	w_shape = W.get_shape()
	output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, w_shape[2]])
	return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keepdims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
def crop_to_shape_v2(data,shape):
    if data.shape[1:3] == shape:
        return data
    h = data.shape[1] - shape[0]
    w = data.shape[2] - shape[1]
    b1 = h//2
    e1 = h - b1
    b2 = w//2
    e2 = w-b2
    return data[:,b1:-e1,b2:-e2]
def get_weights(shape, name, horizontal, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_y = shape[0]//2
        filter_mid_x = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
            if horizontal:
                # All rows after center must be zero
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
                # All columns after center in center row must be zero
                mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.0
            else:
                if mask == 'a':
                    # In the first layer, can ONLY access pixels above it
                    mask_filter[filter_mid_y:, :, :, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    mask_filter[filter_mid_y+1:, :, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else:
            mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.
            mask_filter[filter_mid_y+1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.
                
        W *= mask_filter 
    return W

def get_bias(shape, name):
    return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer)

def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

class GatedCNN():
    def __init__(self, W_shape, fan_in, horizontal, gated=True, payload=None, mask=None, activation=True, conditional=None, conditional_image=None):
        self.fan_in = fan_in
        in_dim = self.fan_in.get_shape()[-1]
        self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]  
        self.b_shape = W_shape[2]

        self.in_dim = in_dim
        self.payload = payload
        self.mask = mask
        self.activation = activation
        self.conditional = conditional
        self.conditional_image = conditional_image
        self.horizontal = horizontal
        
        if gated:
            self.gated_conv()
        else:
            self.simple_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, "v_W", self.horizontal, mask=self.mask)
        W_g = get_weights(self.W_shape, "h_W", self.horizontal, mask=self.mask)

        b_f_total = get_bias(self.b_shape, "v_b")
        b_g_total = get_bias(self.b_shape, "h_b")
        if self.conditional is not None:
            h_shape = int(self.conditional.get_shape()[1])
            V_f = get_weights([h_shape, self.W_shape[3]], "v_V", self.horizontal)
            b_f = tf.matmul(self.conditional, V_f)
            V_g = get_weights([h_shape, self.W_shape[3]], "h_V", self.horizontal)
            b_g = tf.matmul(self.conditional, V_g)

            b_f_shape = tf.shape(b_f)
            b_f = tf.reshape(b_f, (b_f_shape[0], 1, 1, b_f_shape[1]))
            b_g_shape = tf.shape(b_g)
            b_g = tf.reshape(b_g, (b_g_shape[0], 1, 1, b_g_shape[1]))

            b_f_total = b_f_total + b_f
            b_g_total = b_g_total + b_g
        if self.conditional_image is not None:
            b_f_total = b_f_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_f")
            b_g_total = b_g_total + tf.layers.conv2d(self.conditional_image, self.in_dim, 1, use_bias=False, name="ci_g")

        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)
       
        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.multiply(tf.tanh(conv_f + b_f_total), tf.sigmoid(conv_g + b_g_total))

    def simple_conv(self):
        W = get_weights(self.W_shape, "W", self.horizontal, mask_mode="standard", mask=self.mask)
        b = get_bias(self.b_shape, "b")
        conv = conv_op(self.fan_in, W)
        if self.activation: 
            self.fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)

    def output(self):
        return self.fan_out 