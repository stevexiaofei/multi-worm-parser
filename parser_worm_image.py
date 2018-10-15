import argparse
import os
import scipy.misc
import numpy as np
import math
from utils import process_config
from model import Singleout_net

from dataprovider import data_provider
import cv2
import tensorflow as tf
cfg= process_config('exp6//config.cfg')
gene = data_provider(cfg)
Color_list=[(220,20,60),(255,0,255),(138,43,226),(0,0,255),(240,248,255),
(0,255,255),(0,255,127),(0,255,0),(255,255,0),(255,165,0),
(255,69,0),(128,0,0),(255,255,255),(188,143,143)]
Color_name=['Crimson','Magenta','BlueViolet','Blue','AliceBlue',
'Cyan','MediumSpringGreen','Lime','Yellow','Orange',
'OrangeRed','Maroon','White','RosyBrown']
def sample_vector(nums,length):
	direction_vectors=[]
	frac = 2*np.pi/nums
	for i in range(nums):
		direction_vectors.append(np.array([math.cos(frac*i),math.sin(frac*i)],dtype=np.float32))
	direction_vectors=length*np.array(direction_vectors)
	return direction_vectors
class parser_worm_image:
	def __init__(self):
		self.direction_vectors= sample_vector(8,4)
		self.sess=tf.Session()
		self.model = Singleout_net(self.sess,cfg,gene,image_size=cfg['fine_size'], batch_size=cfg['batch_size'],
                   output_size=cfg['fine_size'], dataset_name=cfg['dataset_name'],
                        checkpoint_dir=cfg['checkpoint_dir'], sample_dir=cfg['sample_dir'])
	def parser_image(self,img):
		sample_point_list=self.generate_seed_points(img)
		plot_img,image_patch = self.get_image_patch(img,sample_point_list)
		for i,it in enumerate(sample_point_list[:14]):
			cv2.circle(plot_img,tuple(it),2,Color_list[i%14],-1)
		cv2.imwrite("mask.jpg",plot_img)
		np.savez('center_points.npz',cps=sample_point_list)
		for i in range(len(image_patch)):
			cv2.imwrite('image_patch\\image_patch_{}.jpg'.format(i),image_patch[i])
		# cv2.imshow('bs',plot_img)
		# k = cv2.waitKey(0) 
		# if k == 27:
			# cv2.destroyAllWindows()
		print(image_patch.dtype,image_patch.shape)
		print(np.unique(image_patch[0]))
		self.model.single_out(image_patch.astype(np.float32))
	def get_image_patch(self,img,center_points_list):
		h,w = img.shape
		plot_img=np.stack([img,img,img],axis=2)
		image_path_list=[]
		i=0
		for point in center_points_list:
			x_min=point[0]-128
			y_min=point[1]-128
			x_max=x_min+256
			y_max=y_min+256
			if x_min >=0:
				x_min_pad = 0 
			else: 
				x_min_pad= -x_min
				x_min=0
			if y_min >=0:
				y_min_pad =0
			else:
				y_min_pad = -y_min
				y_min =0 
			if x_max <=w:
				x_max_pad =0
			else:
				x_max_pad =x_max-w
				x_max=w
			if y_max<=h:
				y_max_pad =0
			else:
				y_max_pad =y_max-h
				y_max=h
			if i<14:
				cv2.rectangle(plot_img,(x_min,y_min),(x_max,y_max),Color_list[i],2)
				i+=1
			image_path_list.append(np.pad(img[y_min:y_max,x_min:x_max],\
									((y_min_pad,y_max_pad),(x_min_pad,x_max_pad)),'constant',constant_values=((0,0),(0,0))))
		return plot_img,np.array(image_path_list)
	def generate_seed_points(self,img):
		(_,cnts, hier) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cv2.contourArea
		all_point_list=[]
		for i,it in enumerate(cnts):
			it = np.squeeze(it,axis=1)
			contour_points=[]
			for j in range(0,len(it),22):
				contour_points.append(it[j])
			all_point_list.append(contour_points)
		sample_point_list=[]
		for i,cps in enumerate(all_point_list):
			for it in cps:
				circle_points = (it +self.direction_vectors).astype(np.int32)
				dist_list =[cv2.pointPolygonTest(cnts[i],tuple(point),True) for point in circle_points]
				index =dist_list.index(max(dist_list))
				sample_point_list.append(circle_points[index])
		return sample_point_list
	
if __name__=='__main__':
	img =cv2.imread('ab1.jpg',0)
	_,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	Parser=parser_worm_image()
	Parser.parser_image(img)


