from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import numpy as np
from six.moves import xrange
import scipy.misc
from progressbar import *
from ops import *
from utils import *	
class Singleout_net(object):
	def __init__(self,sess,cfg,data_provider, image_size=256,
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
		self.cfg= cfg
		self.output_class=1024
		self.data_provider= data_provider
		self.is_grayscale = (input_c_dim == 1)
		self.batch_size = batch_size
		self.image_size = image_size
		self.sample_size = sample_size
		self.output_size = output_size
		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')
		self.d_bn4 = batch_norm(name='d_bn4')
		self.d_bn5 = batch_norm(name='d_bn5')
		
		self.input_c_dim = input_c_dim
		self.output_c_dim = output_c_dim

		self.L1_lambda = L1_lambda
		self.build_model()
		t_vars = tf.trainable_variables()
		restore_var = list(filter(lambda x: 'discriminator' not in x.name, t_vars))
		
		#variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
		self.model_path = os.path.join(cfg['exp_name'],'checkpoint')
		self.saver = tf.train.Saver(restore_var)
		self.saver_all= tf.train.Saver()
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
	def build_model(self):
		self.input_image = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name='input_image')
		self.output_image = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.output_c_dim],
                                        name='output_image')
		
		self.weight_mask = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.output_c_dim],
                                        name='weight_mask')
		
		if self.cfg['use_aux_cls']:
			self.label =tf.placeholder(tf.int32,[None,self.output_class],name='labels')
			encode,logits = self.encoder(self.input_image)
			softmax = tf.nn.softmax(logits)
			self.logits=logits
			self.pred = tf.argmax(softmax,1)
			self.cls_loss=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label,logits=logits))
			self.correct_pred = tf.equal(tf.argmax(softmax,1),tf.argmax(self.label,1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
		else:
			encode = self.encoder(self.input_image)
		
		if self.cfg['use_mid_supervise']:
			self.midout_image = tf.placeholder(tf.float32,
                                        [None, self.image_size//4, self.image_size//4,
                                         self.output_c_dim],
                                        name='midout_image')
			self.output,self.mid_output = self.decoder(encode)
			self.diff_loss= -tf.reduce_mean(self.weight_mask*self.output_image*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))+\
				 -tf.reduce_mean(self.weight_mask*(1-self.output_image)*tf.log(tf.clip_by_value(1-self.output,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.midout_image)*tf.log(tf.clip_by_value(1-self.mid_output,1e-10,1.0)))+\
				 -tf.reduce_mean(self.midout_image*tf.log(tf.clip_by_value(self.mid_output,1e-10,1.0)))
		else:
			self.output = self.decoder(encode)
			self.diff_loss= -tf.reduce_mean(self.weight_mask*self.output_image*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))+\
				 -tf.reduce_mean(self.weight_mask*(1-self.output_image)*tf.log(tf.clip_by_value(1-self.output,1e-10,1.0)))
		
		self.crossentropy_loss= -tf.reduce_mean(self.output_image*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.output_image)*tf.log(tf.clip_by_value(1-self.output,1e-10,1.0)))
		
		if self.cfg['use_gan']:
			self.real = tf.concat([self.input_image, self.output_image], 3)
			self.fake = tf.concat([self.input_image, self.output], 3)
			self.D, self.D_logits = self.discriminator(self.real, reuse=False)
			self.D_, self.D_logits_ = self.discriminator(self.fake, reuse=True)
			self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
			self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
			self.d_loss = self.d_loss_real + self.d_loss_fake
	 
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
				+ self.L1_lambda * self.diff_loss
			
		self.loss = 10*self.diff_loss
		if self.cfg['use_aux_cls']:
			self.loss += self.cls_loss 
			self.g_loss += self.cls_loss
			
		t_vars = tf.trainable_variables()
		self.t_vars=t_vars
		self.de_vars = [var for var in t_vars if 'de_' in var.name]
		self.cls_vars = [var for var in t_vars if 'encoder_' in var.name]
		self.discri_vars = [var for var in t_vars if 'discriminator' in var.name]
		for it in self.cls_vars:
			print(it.name)
		self.decoder_vars = [var for var in t_vars if 'decoder_' in var.name]
		
	def train(self, cfg):
		if self.cfg['use_aux_cls']:
			cls_optim = tf.train.AdamOptimizer(cfg['lr'], beta1=cfg['beta1']) \
                          .minimize(self.cls_loss, var_list=self.cls_vars)
		de_optim = tf.train.AdamOptimizer(cfg['lr'], beta1=cfg['beta1']) \
                           .minimize(self.loss, var_list=self.t_vars)
		if cfg['use_gan']:
			d_optim = tf.train.AdamOptimizer(cfg['lr'], beta1=cfg['beta1']) \
							   .minimize(self.d_loss, var_list=self.discri_vars)	
			g_optim = tf.train.AdamOptimizer(cfg['lr'], beta1=cfg['beta1']) \
							   .minimize(self.g_loss, var_list=self.de_vars)		
						   
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		best_acc = 0.
		if cfg['pre_train']:
			for i in range(1000):
				all_loss=0
				for j in range(100):
					mini_batch = self.data_provider(self.batch_size)
					feed = self.get_minibatch(mini_batch)
					_, cls_loss = self.sess.run([cls_optim, self.cls_loss],
												   feed_dict=feed)
					all_loss+=cls_loss
					pred=self.logits.eval(feed)

				print('cls_loss:',all_loss)
			print('pre-train finished! ')
		if cfg['load_model']:
			self.load()
		validate_acc=[]
		validate_xe_loss=[]
		for epoch in xrange(cfg['epoch']):
			batch_idxs =1000
			progress = ProgressBar()
			begin=time.time()
			for idx in progress(xrange(0, batch_idxs)):
				mini_batch = self.data_provider(self.batch_size)
				feed = self.get_minibatch(mini_batch)
				if cfg['use_gan']:
					_, g_loss = self.sess.run([g_optim, self.g_loss],feed_dict=feed)
					_, g_loss = self.sess.run([g_optim, self.g_loss],feed_dict=feed)
					_, d_loss = self.sess.run([d_optim, self.d_loss],feed_dict=feed)
				else:
					_, loss = self.sess.run([de_optim, self.loss],feed_dict=feed)
				if idx%100==0:
					self.sampler(epoch,idx)
				if idx%500==0:
					acc,xe_loss = self.validate(epoch)
					print('acc',acc)
					validate_acc.append(acc)
					validate_xe_loss.append(xe_loss)
					if acc>best_acc:
						self.save()
			print('epoch time:',(time.time()-begin)/60,'min')
			np.savez(os.path.join(self.cfg['exp_name'],'validation_acc.npz'),acc= validate_acc,loss=validate_xe_loss)
	def get_minibatch(self,mini_batch): 
		input_image_batch, output_image_batch,w_mask,midout_image_batch, label_batch = mini_batch
		feed= {self.input_image: input_image_batch,\
				self.output_image:output_image_batch,
				self.weight_mask:w_mask}
		if self.cfg['use_aux_cls']:
			feed[self.label]=label_batch
		if self.cfg['use_mid_supervise']:
			feed[self.midout_image]= midout_image_batch
		return feed
	def single_out(self,image_batch):
		image_batch -= np.amin(image_batch)
		image_batch /= np.amax(image_batch)
		image_batch = image_batch[...,np.newaxis]
		num = len(image_batch)
		num_batch = num//4
		remain = num%4
		output_image=[]
		self.load()
		prob_list = []
		print('num_batch',num_batch)
		for i in range(num_batch):
			output_image_batch = self.sess.run(self.output,feed_dict={self.input_image:image_batch[i*4:(i+1)*4]})
			if self.cfg['use_gan']:
				prob =self.sess.run(self.D_, feed_dict={self.input_image:image_batch[i*4:(i+1)*4]} )
				#print(prob,len(prob),len(image_batch[i*4:(i+1)*4]))
				prob_list+=list(prob)
			output_image += list(output_image_batch)
		print(len(prob_list))
		for i in range(len(output_image)):
			save_image= np.squeeze(output_image[i])
			save_image-= np.amin(save_image)
			save_image /=np.amax(save_image)
			save_image = save_image*255
			cv2.imwrite('singleout_image\\image_patch_{}.jpg'.format(i),save_image.astype(np.uint8))
		# for i, it in enumerate(prob_list):
			# print(i,it)
	def sampler(self,epoch,idx):
		mini_batch = self.data_provider(self.batch_size)
		feed = self.get_minibatch(mini_batch)
		output_image = self.output.eval(feed)
		save_image([mini_batch[0],mini_batch[1],output_image],os.path.join(self.cfg['exp_name'],'sample',
															'%s_%s.jpg'%(str(epoch),str(idx))))
		output_image = (np.sign(output_image-0.5)+1)/2
		accuracy = np.equal(mini_batch[1],output_image).astype(np.float32).mean()
		print('epoch ',epoch,\
					' idx ',idx,
					' accuracy',accuracy)
	def validate(self,num_epoch,num_iter=100):
		state = np.random.get_state()
		np.random.seed(111)
		mean_acc = 0.
		mean_xe_loss=0.
		for input_image_batch, output_image_batch in self.data_provider.generator_test(num_iter):
			feed = {self.input_image: input_image_batch,\
					self.output_image:output_image_batch}
			output_image,xe_loss = self.sess.run([self.output,self.crossentropy_loss],feed_dict=feed)

			output_image = (np.sign(output_image-0.5)+1)/2
			accuracy = np.equal(output_image_batch,output_image).astype(np.float32).mean()
			mean_acc+=accuracy
			mean_xe_loss +=xe_loss
		np.random.set_state(state)
		return mean_acc/num_iter, mean_xe_loss/num_iter
	def encoder(self,input):
		self.downside_layers={}
		with tf.variable_scope('de_encoder_preprocessing'):
			pad1=tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
			conv1 = self._conv_bn_relu(pad1, filters= 8, kernel_size = 3, strides = 1, name = 'conv_channel_to_64')
			in_node = self._residual_pool(conv1, numOut = 16, name = 'r1')
		num_filters = 16
		
		with tf.variable_scope('de_encoder_downside'):
			for i in range(1,7):
				down = self._conv_bn_relu(in_node,filters= num_filters, kernel_size=2, strides =2, name ='conv_%s'%(str(i)))
				#128*128
				pool = self._residual_pool(down,numOut= num_filters, name='rpl_%s'%(str(i)))
				in_node = pool
				num_filters = num_filters*2
				self.downside_layers['pool_%s'%str(i)]=pool#(64,64,32)

		if self.cfg['use_aux_cls']:
			out = linear(tf.reshape(pool, [self.batch_size, -1]), self.output_class, 'encoder_linear')
			return tf.nn.relu(pool),out
		else:
			return tf.nn.relu(pool)
	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator") as scope:
            # image is 256 x 256 x (input_c_dim + output_c_dim)
			if reuse:
				tf.get_variable_scope().reuse_variables()
			else:
				assert tf.get_variable_scope().reuse == False
			h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
			# h0 is (128 x 128 x self.df_dim)
			h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
			# h1 is (64 x 64 x self.df_dim*2)
			h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
			# h2 is (32x 32 x self.df_dim*4)
			h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
			# h3 is (16x 16 x self.df_dim*8)
			h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*8, name='d_h4_conv')))
			h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*8, name='d_h5_conv')))
			# h3 is (4 x 4 x self.df_dim*8)
			#print('h5 ',h5.get_shape())
			h6 = linear(tf.reshape(h5, [-1, 8192]), 1, 'd_h3_lin')
		return tf.nn.sigmoid(h6), h6
	def decoder(self,input):
		self.upside_layers=[]
		with tf.variable_scope('de_decoder_upside'):
			num_filters= 256
			for i in range(1,7):
				up= self.up_sample(input,num_filters,name='ups_%s'%(str(i)))
				if i<= self.cfg['connect_layers']:
					up=self._conv_bn_relu(tf.concat([up,self.downside_layers['pool_%d'%(6-i)]],3),num_filters,kernel_size=3,pad='SAME',name='concat_%d'%(i))
				pool= self._residual_pool(up,numOut= num_filters,name='up_%d'%(i))
				if i<= self.cfg['dropout_layers']:
					pool= tf.nn.dropout(pool, 0.5)
				num_filters = num_filters/2
				self.upside_layers.append(pool)
				input = pool

		with tf.variable_scope('de_decoder_pixelCNN'):
			v_stack_in=pool
			h_stack_in=pool
			for i in range(12):
				filter_size = 5 if i > 0 else 7
				mask = 'b' if i > 0 else 'a'
				residual = True if i > 0 else False
				conditional_image =None #self.input_image if i<3 else None
				i = str(i)
				with tf.variable_scope("v_stack"+i):
					v_stack =GatedCNN([filter_size, filter_size, 16], v_stack_in, False, mask=mask,conditional_image=conditional_image).output()
					v_stack_in = v_stack
				with tf.variable_scope("v_stack_1"+i):
					v_stack_1 = GatedCNN([1, 1, 16], v_stack_in, False, gated=False, mask=None,conditional_image=conditional_image).output()
				with tf.variable_scope("h_stack"+i):
					h_stack = GatedCNN([filter_size , filter_size, 16], h_stack_in, True, payload=v_stack_1, mask=mask,conditional_image=conditional_image).output()
				with tf.variable_scope("h_stack_1"+i):
					h_stack_1 = GatedCNN([1, 1, 16], h_stack, True, gated=False, mask=None,conditional_image=conditional_image).output()
					if residual:
						h_stack_1 += h_stack_in # Residual connection
					h_stack_in = h_stack_1
		with tf.variable_scope('de_decoder_output'):
			_out = h_stack_in if self.cfg['use_pixcnn'] else pool
			_out =tf.pad(h_stack_in, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
			out= self._conv(_out, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv') 
			if self.cfg['use_mid_supervise']:
				mid_out = tf.pad(self.upside_layers[3],[[0,0],[1,1],[1,1],[0,0]], name='pad2')#todo
				mid_out = self._conv(mid_out, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'mid_out')
		if self.cfg['use_mid_supervise']:
			return tf.nn.sigmoid(out),tf.nn.sigmoid(mid_out)
		else:
			return tf.nn.sigmoid(out)
	def save(self):
		model_name = "singleout.model"

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)

		self.saver_all.save(self.sess,
                        os.path.join(self.model_path, model_name),
                        )

	def load(self):
		checkpoint_dir = self.model_path
		if self.cfg['load_model_dir'] is not None:
			checkpoint_dir = self.cfg['load_model_dir']
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			print(" [*] Reading checkpoint... SUCCESSFUL") 
			return True
		else:
			print(" [*] Reading checkpoint... FAILED") 
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
			
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
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
			
			h_deconv = tf.nn.relu(deconv2d(inputs, kernel, pool_size))
			print(h_deconv.get_shape())
		return h_deconv
		
		
	
	
	
	