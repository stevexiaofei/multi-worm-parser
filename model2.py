from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import cv2
import numpy as np
from six.moves import xrange
import scipy.misc
from ops import *
from utils import *
import layers 
def save_image(imgs1,imgs2,imgs3,path):
	imgs1=imgs1.reshape([-1,256])
	imgs2=imgs2.reshape([-1,256])
	imgs3=imgs3.reshape([-1,256])
	#imgs4=imgs4.reshape([-1,256])
	scipy.misc.imsave(path, np.concatenate([imgs1,imgs2,imgs3],axis=1))
class Singleout_net(object):
	def __init__(self,sess,data_provider, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=1, output_c_dim=1, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):
		"""

			Args:
				sess: TensorFlow session
				batch_size: The size of batch. Should be specified before training.
				output_size: (optional) The resolution in pixels of the images. [256]
				gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
				df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
				input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
				output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
		"""
		self.sess = sess
		self.output_class=1024
		self.data_provider= data_provider
		self.is_grayscale = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = sample_size
		self.output_size = output_size
		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim

		self.L1_lambda = L1_lambda
		self.build_model()
		self.saver = tf.train.Saver()
	def build_model(self):
		self.input_image = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name='input_image')
		self.output_image = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.output_c_dim],
                                        name='output_image')
		self.label =tf.placeholder(tf.int32,[self.batch_size,self.output_class],name='labels')
		encode,logits = self.encoder(self.input_image)
		self.output = self.decoder(encode)
		
		self.diff_loss= -tf.reduce_mean(self.output_image*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.output_image)*tf.log(tf.clip_by_value(1-self.output,1e-10,1.0)))
		
		#logits = self.classify(encode)
		softmax = tf.nn.softmax(logits)
		self.logits=logits
		#index_matrix=tf.transpose(tf.stack([tf.range(self.batch_size),self.label],axis=0),perm=[1,0])
		self.pred = tf.argmax(softmax,1)
		self.cls_loss=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,logits=logits))
		
		self.loss = self.cls_loss + 10*self.diff_loss
		self.correct_pred = tf.equal(tf.argmax(softmax,1),tf.argmax(self.label,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
		
		t_vars = tf.trainable_variables()
		self.t_vars=t_vars
		self.de_vars = [var for var in t_vars if 'de_' in var.name]
		self.cls_vars = [var for var in t_vars if 'encoder_' in var.name]
		for it in self.cls_vars:
			print(it.name)
		self.decoder_vars = [var for var in t_vars if 'decoder_' in var.name]
		
	def train(self, args):
		cls_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.cls_loss, var_list=self.cls_vars)
		de_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                           .minimize(self.loss, var_list=self.t_vars)
		decoder_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                           .minimize(self.diff_loss, var_list=self.decoder_vars)	
						  
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		# for i in range(1000):
			# all_loss=0
			# for j in range(100):
				# input_image_batch, output_image_batch, label_batch = self.data_provider(self.batch_size)
				# _, cls_loss = self.sess.run([cls_optim, self.cls_loss],
                                               # feed_dict={self.input_image: input_image_batch,\
																  # self.output_image:output_image_batch,
															  # self.label:label_batch})
				
				# all_loss+=cls_loss
				# pred=self.logits.eval({self.input_image: input_image_batch,\
																  # self.output_image:output_image_batch,
															  # self.label:label_batch})

			# print('cls_loss:',all_loss)
		self.load(args.checkpoint_dir)
		for epoch in xrange(args.epoch):
			batch_idxs =1000
			for idx in xrange(0, batch_idxs):
				input_image_batch, output_image_batch, label_batch = self.data_provider(self.batch_size)
				_, loss = self.sess.run([de_optim, self.loss],
                                               feed_dict={self.input_image: input_image_batch,\
																  self.output_image:output_image_batch,
																  self.label:label_batch})
						  
				if idx%100==0:
					input_image_batch, output_image_batch, label_batch = self.data_provider(self.batch_size)
					output_image = self.output.eval({self.input_image: input_image_batch,\
																  self.output_image:output_image_batch,
																  self.label:label_batch})
					accuracy = self.accuracy.eval({self.input_image: input_image_batch,\
																  self.output_image:output_image_batch,
															  self.label:label_batch})
					print('epoch ',epoch,' idx ',idx,' loss: ',loss)	
					print('accuracy',accuracy)
					output_image=output_image[...,0]
					input_image_batch=input_image_batch[...,0]
					output_image_batch=output_image_batch[...,0]
					save_image(input_image_batch,output_image_batch,output_image,'%s_%s.jpg'%(str(epoch),str(idx)))

			self.save(args.checkpoint_dir)		
	def encoder(self,input):
		with tf.variable_scope('de_encoder_preprocessing'):
			pad1=tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
			conv1 = self._conv_bn_relu(pad1, filters= 8, kernel_size = 3, strides = 1, name = 'conv_channel_to_64')
			in_node = self._residual_pool(conv1, numOut = 16, name = 'r1')
		
		with tf.variable_scope('de_encoder_downside'):
			down_2 = self._conv_bn_relu(in_node,filters=16,kernel_size=2, strides =2, name ='conv_8to16')
			#128*128
			pool_2 = self._residual_pool(down_2,numOut=16,name='rpl_2')
			down_3 = self._conv_bn_relu(pool_2,filters=32,kernel_size=2, strides =2, name ='conv_16to32')
			#64*64
			pool_3 = self._residual_pool(down_3,numOut=32,name='rpl_3')
			down_4 = self._conv_bn_relu(pool_3,filters=64,kernel_size=2, strides =2, name ='conv_32to64')
			#32*32
			pool_4 = self._residual_pool(down_4,numOut=64,name='rpl_4')
			down_5 = self._conv_bn_relu(pool_4,filters=128,kernel_size=2, strides =2, name ='conv_64to128')
			#16*16
			pool_5 = self._residual_pool(down_5,numOut=128,name='rpl_5')
			down_6 = self._conv_bn_relu(pool_5,filters=256,kernel_size=2, strides =2, name ='conv_128to256')
			#8*8
			pool_6 = self._residual_pool(down_6,numOut=256,name='rpl_6')
			down_7 = self._conv_bn_relu(pool_6,filters=512,kernel_size=2, strides =2, name ='conv_256to512')
			#4*4
			pool_7 = self._residual_pool(down_7,numOut=512,name='rpl_7')
			out = linear(tf.reshape(pool_7, [self.batch_size, -1]), self.output_class, 'encoder_linear')
		return tf.nn.relu(pool_7),out
	def classify(self,input):
		out = linear(tf.reshape(input, [self.batch_size, -1]), self.output_class, 'encoder_linear')
		return tf.nn.sigmoid(out)
	def decoder(self,input):
		with tf.variable_scope('de_decoder_upside'):
			up_2 = self.up_sample(input,256,name='ups_2')
			#8*8
			pool_2 = self._residual_pool(up_2,numOut=256,name='up_2')
			pool_2 = tf.nn.dropout(pool_2, 0.5)
			up_3 = self.up_sample(pool_2,128,name='ups_3')
			#16*16
			pool_3 = self._residual_pool(up_3,numOut=128,name='up_3')
			pool_3 = tf.nn.dropout(pool_3, 0.5)
			up_4 = self.up_sample(pool_3,64,name='ups_4')
			#32*32
			pool_4 = self._residual_pool(up_4,numOut=64,name='up_4')
			up_5 = self.up_sample(pool_4,32,name='ups_5')
			#64*64
			pool_5 = self._residual_pool(up_5,numOut=32,name='up_5')
			up_6 = self.up_sample(pool_5,16,name='ups_6')
			#128*128
			pool_6 = self._residual_pool(up_6,numOut=16,name='up_6')
			up_7 = self.up_sample(pool_6,16,name='ups_7')
			#256*256
			pool_7 = self._residual_pool(up_7,numOut=16,name='up_7')
		with tf.variable_scope('de_decoder_pixelCNN'):
			v_stack_in=pool_7
			h_stack_in=pool_7
			for i in range(12):
				filter_size = 5 if i > 0 else 7
				mask = 'b' if i > 0 else 'a'
				residual = True if i > 0 else False
				i = str(i)
				with tf.variable_scope("v_stack"+i):
					v_stack =layers.GatedCNN([filter_size, filter_size, 16], v_stack_in, False, mask=mask).output()
					v_stack_in = v_stack
				with tf.variable_scope("v_stack_1"+i):
					v_stack_1 = layers.GatedCNN([1, 1, 16], v_stack_in, False, gated=False, mask=None).output()
				with tf.variable_scope("h_stack"+i):
					h_stack = layers.GatedCNN([filter_size , filter_size, 16], h_stack_in, True, payload=v_stack_1, mask=mask).output()
				with tf.variable_scope("h_stack_1"+i):
					h_stack_1 = layers.GatedCNN([1, 1, 16], h_stack, True, gated=False, mask=None).output()
					if residual:
						h_stack_1 += h_stack_in # Residual connection
					h_stack_in = h_stack_1
		with tf.variable_scope('de_decoder_output'):
			_out =tf.pad(h_stack_in, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
			out= self._conv(_out, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
		return tf.nn.sigmoid(out) 
	def save(self, checkpoint_dir):
		model_name = "pix2pix.model"
		model_dir = "%s_%s" % ( self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        )

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s" % (self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
	def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
		""" Spatial Convolution (CONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.variable_scope(name):
			# Kernel for convolution, Xavier Initialisation
			kernel = tf.get_variable('weights',[kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			return conv
	def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
		""" Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
		with tf.variable_scope(name):
			kernel = tf.get_variable('weights',[kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )
			
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu,updates_collections=None,scale=True)
			#if self.w_summary:
			#	with tf.device('/cpu:0'):
			#		tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm
	def _conv_block(self, inputs, numOut, name = 'conv_block'):
		""" Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
		with tf.variable_scope(name):
			with tf.variable_scope('norm_1'):
				norm_1 =  tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu,updates_collections=None,scale=True)
				conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
			with tf.variable_scope('norm_2'):
				norm_2 =  tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu,updates_collections=None,scale=True)
				pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
				conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
			with tf.variable_scope('norm_3'):
				norm_3 =  tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu,updates_collections=None,scale=True)
				conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
		return conv_3
	def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
		""" Skip Layer
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
		with tf.variable_scope(name):
			if inputs.get_shape().as_list()[3] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv
	def _residual(self, inputs, numOut, modif = False, name = 'residual_block'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.variable_scope(name):
			convb = self._conv_block(inputs, numOut)
			skipl = self._skip_layer(inputs, numOut)
			if modif:
				return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
			else:
				return tf.add_n([convb, skipl], name = 'res_block')
	def _bn_relu(self, inputs):
		norm =  tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu,updates_collections=None,scale=True)
		return norm
	def _pool_layer(self, inputs, numOut, name = 'pool_layer'):
		with tf.variable_scope(name):
			bnr_1 = self._bn_relu(inputs)
			pool = tf.contrib.layers.max_pool2d(bnr_1,[2,2],[2,2],padding='VALID')
			pad_1 = tf.pad(pool, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_1 = self._conv(pad_1, numOut, kernel_size=3, strides=1, name='p_conv_1')
			bnr_2 = self._bn_relu(conv_1)
			pad_2 = tf.pad(bnr_2, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_2 = self._conv(pad_2, numOut, kernel_size=3, strides=1, name='p_conv_2')
			upsample = tf.image.resize_nearest_neighbor(conv_2, tf.shape(conv_2)[1:3]*2, name = 'upsampling')
		return upsample
	def _residual_pool(self, inputs, numOut, name = 'residual_pool'):
		with tf.variable_scope(name):
			return tf.add_n([self._conv_block(inputs, numOut,name='rp_conv1'), self._skip_layer(inputs, numOut,name='rp_conv2'), self._pool_layer(inputs, numOut,name='rp_conv3')])
	def _lin(self, inputs, numOut, name = 'lin'):
		with tf.variable_scope(name):
			l = self._conv(inputs, filters = numOut, kernel_size = 1, strides = 1)
		return self._bn_relu(l)
	def up_sample(self,inputs,numOut,pool_size = 2,name = 'upsample'):
		with tf.variable_scope(name):
			kernel = tf.get_variable('weights',[pool_size,pool_size, numOut, inputs.get_shape().as_list()[3]],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )

			#wd = weight_variable_devonc([pool_size, pool_size, numOut// 2, numOut], stddev)
			#bd = bias_variable([features // 2])
			print(inputs.get_shape(),kernel.get_shape())
			
			h_deconv = tf.nn.relu(layers.deconv2d(inputs, kernel, pool_size))
			print(h_deconv.get_shape())
		return h_deconv
		
		
	
	
	
	