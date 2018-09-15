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
def save_image(imgs1,imgs2,imgs3,imgs4,path):
	imgs1=imgs1.reshape([-1,256])
	imgs2=imgs2.reshape([-1,256])
	imgs3=imgs3.reshape([-1,256])
	imgs4=imgs4.reshape([-1,256])
	scipy.misc.imsave(path, np.concatenate([imgs1,imgs2,imgs3,imgs4],axis=1))
class pix2pix(object):
	def __init__(self, sess,data_provider, image_size=256,
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

        # batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
		self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn_e2 = batch_norm(name='g_bn_e2')
		self.g_bn_e3 = batch_norm(name='g_bn_e3')
		self.g_bn_e4 = batch_norm(name='g_bn_e4')
		self.g_bn_e5 = batch_norm(name='g_bn_e5')
		self.g_bn_e6 = batch_norm(name='g_bn_e6')
		self.g_bn_e7 = batch_norm(name='g_bn_e7')
		self.g_bn_e8 = batch_norm(name='g_bn_e8')

		self.g_bn_d1 = batch_norm(name='g_bn_d1')
		self.g_bn_d2 = batch_norm(name='g_bn_d2')
		self.g_bn_d3 = batch_norm(name='g_bn_d3')
		self.g_bn_d4 = batch_norm(name='g_bn_d4')
		self.g_bn_d5 = batch_norm(name='g_bn_d5')
		self.g_bn_d6 = batch_norm(name='g_bn_d6')
		self.g_bn_d7 = batch_norm(name='g_bn_d7')

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.build_model()

	def build_model(self):
		self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

		self.real_B = self.real_data[:, :, :, :self.input_c_dim]
		self.real_A = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

		self.fake_B,self.fake_B1 = self.generator_v1(self.real_A)

		self.real_AB = tf.concat([self.real_A, self.real_B], 3)
		self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
		self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
		self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

		self.fake_B_sample_1,self.fake_B_sample = self.sampler_v1(self.real_A)

		self.d_sum = tf.summary.histogram("d", self.D)
		self.d__sum = tf.summary.histogram("d_", self.D_)
		self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
		self.diff_loss= -tf.reduce_mean(self.real_B*tf.log(tf.clip_by_value(self.fake_B,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.real_B)*tf.log(tf.clip_by_value(1-self.fake_B,1e-10,1.0)))+\
				 -tf.reduce_mean(self.real_B*tf.log(tf.clip_by_value(self.fake_B1,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.real_B)*tf.log(tf.clip_by_value(1-self.fake_B1,1e-10,1.0)))
		#tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * self.diff_loss

		self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
		self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
		for it in self.d_vars:
			print(it.name)
		for it in self.g_vars:
			print(it.name)
		self.saver = tf.train.Saver()
		self.init_op= tf.global_variables_initializer()

	def load_random_samples(self):
		data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
		sample = [load_data(sample_file) for sample_file in data]

		if (self.is_grayscale):
			sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
		else:
			sample_images = np.array(sample).astype(np.float32)
		return sample_images

	def sample_model(self, sample_dir, epoch, idx):
		sample_images = self.data_provider(self.batch_size)
		samples1,samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample_1,self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
		imgs=sample_images[:,:,:,0]
		labels=sample_images[:,:,:,1]
		samples=samples[:,:,:,0]
		samples-=np.amin(samples)
		samples/=np.amax(samples)
		samples1=samples1[:,:,:,0]
		samples1-=np.amin(samples1)
		samples1/=np.amax(samples1)
		save_image(imgs,labels,samples,samples1,
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
		print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

	def train(self, args):
		"""Train pix2pix"""


		d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		self.g_sum = tf.summary.merge([self.d__sum,
            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
		self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
		self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

		counter = 1
		start_time = time.time()

		for epoch in xrange(args.epoch):
            #data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            #np.random.shuffle(data)
			batch_idxs =1000 #min(len(data), args.train_size) // self.batch_size

			for idx in xrange(0, batch_idxs):
                #batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                #batch = [load_data(batch_file) for batch_file in batch_files]
                #if (self.is_grayscale):
                #    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                #else:
                #    batch_images = np.array(batch).astype(np.float32)
				batch_images = self.data_provider(self.batch_size)
                # Update D network
				_, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={ self.real_data: batch_images })
				self.writer.add_summary(summary_str, counter)

                # Update G network
				_, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
				self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				_, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
				self.writer.add_summary(summary_str, counter)
				_, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={ self.real_data: batch_images })
				self.writer.add_summary(summary_str, counter)
				errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
				errD_real = self.d_loss_real.eval({self.real_data: batch_images})
				errG = self.g_loss.eval({self.real_data: batch_images})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

				if np.mod(counter, 100) == 1:
					self.sample_model(args.sample_dir, epoch, idx)

				if np.mod(counter, 500) == 2:
					self.save(args.checkpoint_dir, counter)

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
			h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

			return tf.nn.sigmoid(h4), h4

	def generator(self, image, y=None):
		with tf.variable_scope("generator") as scope:

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
			e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
			e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
			e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
			e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
			e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
			e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
			e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

			self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
			d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
			d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

			self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
			d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
			d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

			self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
			d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
			d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

			self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
			d4 = self.g_bn_d4(self.d4)
			d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

			self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
			d5 = self.g_bn_d5(self.d5)
			d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

			self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
			d6 = self.g_bn_d6(self.d6)
			d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

			self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
			d7 = self.g_bn_d7(self.d7)
			d7 = tf.concat([d7, e1], 3) 
            # d7 is (128 x 128 x self.gf_dim*1*2)

			self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

			return tf.nn.sigmoid(self.d8)

	def sampler(self, image, y=None):

		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()

			s = self.output_size
			s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
			e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
			e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
			e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
			e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
			e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
			e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
			e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
			e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

			self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
			d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
			d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

			self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
			d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
			d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

			self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
			d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
			d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

			self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
			d4 = self.g_bn_d4(self.d4)
			d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

			self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
			d5 = self.g_bn_d5(self.d5)
			d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

			self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
			d6 = self.g_bn_d6(self.d6)
			d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

			self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
			d7 = self.g_bn_d7(self.d7)
			d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

			self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

			return tf.nn.sigmoid(self.d8)

	def save(self, checkpoint_dir, step):
		model_name = "pix2pix.model"
		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoint...")

		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
	def single_out(self,image_patch):
		num_patchs=len(image_patch)
		print(image_patch.shape)
		single_out_img=[]
		
		num_batch=num_patchs//self.batch_size
		remains= num_patchs%self.batch_size
		for i in range(num_batch):
			begin= i*self.batch_size
			samples = self.sess.run(self.fake_B_sample,
                feed_dict={self.real_data:image_patch[begin:begin+self.batch_size]}
            )
			for it in samples:
				single_out_img.append(it)
		if remains!=0:
			samples = self.sess.run(self.fake_B_sample,
                feed_dict={self.real_data:image_patch[num_patchs-self.batch_size:num_patchs]}
            )
			for i in range(remains):
				single_out_img.append(samples[-(i+1)])
		single_out_img=np.array(single_out_img)
		single_out_img-=np.amin(single_out_img)
		single_out_img/=np.amax(single_out_img)
		single_out_img= (single_out_img*255).astype(np.uint8)
		for i,it in enumerate(single_out_img):
			cv2.imwrite('singleout_image\\single_out_{}.jpg'.format(i),it)
	def test(self, args):
		"""Test pix2pix"""
		start_time = time.time()
		def to_255(img):
			img=img*255
			return img.astype(np.uint8)
		for i in xrange(100):
			sample_images = self.data_provider(self.batch_size)
			idx = i+1
			print("sampling image ", i)
			samples1,samples = self.sess.run([self.fake_B_sample_1,self.fake_B_sample],
                feed_dict={self.real_data: sample_images}
            )
			imgs=sample_images[:,:,:,0]
			labels=sample_images[:,:,:,1]
			labels_ =to_255(labels)
			samples=samples[:,:,:,0]
			samples-=np.amin(samples)
			samples/=np.amax(samples)
			#ret,samples_=cv2.threshold(to_255(samples)[0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			
			#(_,labels_cnts, hier) = cv2.findContours(labels_[0], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
			#(_,samples_cnts, hier)=cv2.findContours(samples_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
			#mask=np.zeros((256,256,3))
			#cv2.drawContours(mask, labels_cnts, -1, (255,255,0), thickness=1)
			#cv2.drawContours(mask, samples_cnts, -1, (255,0,255), thickness=1)
			#cv2.imwrite("img{}.jpg".format(i),mask)
			samples1=samples1[:,:,:,0]
			samples1-=np.amin(samples1)
			samples1/=np.amax(samples1)
			save_image(imgs,labels,samples,samples1,
						'./{}/test_{:04d}.png'.format('test_sample',i))
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
	def _attention_iter(self, inputs, lrnSize, itersize, name = 'attention_iter'):
		with tf.variable_scope(name):
			numIn = inputs.get_shape().as_list()[3]
			padding = np.floor(lrnSize/2)
			pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
			U = self._conv(pad, filters=1, kernel_size=3, strides=1)
			pad_2 = tf.pad(U, np.array([[0,0],[padding,padding],[padding,padding],[0,0]]))
			sharedK = tf.get_variable('shared_weights',[lrnSize,lrnSize, 1, 1],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )
			
			Q = []
			C = []
			for i in range(itersize):
				if i ==0:
					conv = tf.nn.conv2d(pad_2, sharedK, [1,1,1,1], padding='VALID', data_format='NHWC')
				else:
					conv = tf.nn.conv2d(Q[i-1], sharedK, [1,1,1,1], padding='SAME', data_format='NHWC')
				C.append(conv)
				Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))
				Q.append(Q_tmp)
			stacks = []
			for i in range(numIn):
				stacks.append(Q[-1]) 
			pfeat = tf.multiply(inputs,tf.concat(stacks, axis = 3) )
		return pfeat
	def _residual_pool(self, inputs, numOut, name = 'residual_pool'):
		with tf.variable_scope(name):
			return tf.add_n([self._conv_block(inputs, numOut,name='rp_conv1'), self._skip_layer(inputs, numOut,name='rp_conv2'), self._pool_layer(inputs, numOut,name='rp_conv3')])
	def _lin(self, inputs, numOut, name = 'lin'):
		with tf.variable_scope(name):
			l = self._conv(inputs, filters = numOut, kernel_size = 1, strides = 1)
		return self._bn_relu(l)
		
	def _rep_residual(self, inputs, numOut, nRep, name = 'rep_residual'):
		with tf.variable_scope(name):
			out = [None]*nRep
			for i in range(nRep):
				if i == 0:
					tmpout = self._residual(inputs,numOut,name='resid%s'%(str(i)))
				else:
					tmpout = self._residual_pool(out[i-1],numOut,name='rpl%s'%(str(i)))
				out[i] = tmpout
			return out[nRep-1]
	def up_sample(self,inputs,numOut,pool_size = 2,name = 'upsample'):
		with tf.variable_scope('upsample'):
			kernel = tf.get_variable('weights',[pool_size,pool_size, numOut, inputs.get_shape().as_list()[3]],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )

			#wd = weight_variable_devonc([pool_size, pool_size, numOut// 2, numOut], stddev)
			#bd = bias_variable([features // 2])
			h_deconv = tf.nn.relu(layers.deconv2d(inputs, kernel, pool_size))
		return h_deconv
	def _hg_mcam(self, inputs, n, numOut, nModual, name = 'mcam_hg'):
		with tf.variable_scope(name):
			#------------Upper Branch
			pool = tf.contrib.layers.max_pool2d(inputs,[2,2],[2,2],padding='VALID')
			up = []
			low = [] 
			for i in range(nModual):
				if i == 0:
					if n>1:
						tmpup = self._rep_residual(inputs, numOut, n -1,name='rpr_%s'%(str(i)))
					else:
						tmpup = self._residual(inputs, numOut,name='up_r_%s'%(str(i)))
					tmplow = self._residual(pool, numOut*2,name='low_r_%s'%(str(i)))
				else:
					if n>1:
						tmpup = self._rep_residual(up[i-1], numOut, n-1,name='rpr_%s'%(str(i)))
					else:
						tmpup = self._residual_pool(up[i-1], numOut,name='up_r_%s'%(str(i)))
					tmplow = self._residual(low[i-1], numOut*2,name='low_r_%s'%(str(i)))
				up.append(tmpup)
				low.append(tmplow)
			
			if n>1:
				low2 = self._hg_mcam(low[-1], n-1, numOut*2, nModual)
			else:
				low2 = self._residual(low[-1], numOut*2,name='low2_r')
			if n<3:
				low2 = tf.nn.dropout(low2, 0.5)
			low3 = self._residual(low2, numOut*2,name='low3_r')
			#up_2 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3]*2, name = 'upsampling')
			up_2 = self.up_sample(low3,numOut)
			concat = tf.concat([up[-1], up_2],axis=3, name = 'concat')
			kernel = tf.get_variable('weights',[1,1, 2*numOut, numOut],initializer=tf.contrib.layers.xavier_initializer(uniform=False) )			
			conv = tf.nn.conv2d(concat, kernel, [1,1,1,1], padding="VALID", data_format='NHWC')
			return conv

	def generator_v1(self, image, y=None):
		with tf.variable_scope("generator1") as scope:
			with tf.variable_scope('g_preprocessing'):
				pad1=tf.pad(image, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				conv1 = self._conv_bn_relu(pad1, filters= 8, kernel_size = 3, strides = 1, name = 'conv_channel_to_64')
				in_node = self._residual_pool(conv1, numOut = 16, name = 'r1')
			with tf.variable_scope('g_unet'):
				in_node=self._hg_mcam(in_node,6,16,1)
			with tf.variable_scope('g_attention'):
				pad1=tf.pad(in_node, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				d2= self._conv(pad1, 8, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
				att = self._attention_iter(d2,5,3)
			with tf.variable_scope('g_output'):
				_out =tf.pad(att, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				out= self._conv(_out, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
			with tf.variable_scope('g_unet2'):
				in_node=self._hg_mcam(att,6,16,1)
			with tf.variable_scope('g_output2'):
				_out2 =tf.pad(in_node, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				out2= self._conv(_out2, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
		return tf.nn.sigmoid(out),tf.nn.sigmoid(out2)
	def sampler_v1(self, image, y=None):
		with tf.variable_scope("generator1") as scope:
			scope.reuse_variables()
			with tf.variable_scope('g_preprocessing'):
				pad1=tf.pad(image, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				conv1 = self._conv_bn_relu(pad1, filters= 8, kernel_size = 3, strides = 1, name = 'conv_channel_to_64')
				in_node = self._residual_pool(conv1, numOut = 16, name = 'r1')
			with tf.variable_scope('g_unet'):
				in_node=self._hg_mcam(in_node,6,16,1)
			with tf.variable_scope('g_attention'):
				pad1=tf.pad(in_node, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				d2= self._conv(pad1, 8, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
				att = self._attention_iter(d2,5,3)
			with tf.variable_scope('g_output'):
				_out =tf.pad(att, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				out= self._conv(_out, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
			with tf.variable_scope('g_unet2'):
				in_node=self._hg_mcam(att,6,16,1)
			with tf.variable_scope('g_output2'):
				_out2 =tf.pad(in_node, [[0,0],[1,1],[1,1],[0,0]], name='pad1')
				out2= self._conv(_out2, self.output_c_dim, kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
		return tf.nn.sigmoid(out),tf.nn.sigmoid(out2)