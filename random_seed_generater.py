import argparse
import os
import scipy.misc
import numpy as np
import math
from model import pix2pix
from dataprovider import gene
import cv2
import tensorflow as tf
Color_list=[(220,20,60),(255,0,255),(138,43,226),(0,0,255),(240,248,255),
(0,255,255),(0,255,127),(0,255,0),(255,255,0),(255,165,0),
(255,69,0),(128,0,0),(255,255,255),(188,143,143)]
Color_name=['Crimson','Magenta','BlueViolet','Blue','AliceBlue',
'Cyan','MediumSpringGreen','Lime','Yellow','Orange',
'OrangeRed','Maroon','White','RosyBrown']
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='test', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint1', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()
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
		self.model = pix2pix(self.sess, gene,image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)
	def parser_image(self,img):
		sample_point_list=self.generate_seed_points(img)
		plot_img,image_patch = self.get_image_patch(img,sample_point_list)
		for i,it in enumerate(sample_point_list[:14]):
			cv2.circle(plot_img,tuple(it),2,Color_list[i%14],-1)
		cv2.imwrite("mask.jpg",plot_img)
		for i in range(len(image_patch)):
			cv2.imwrite('image_patch\\image_patch_{}.jpg'.format(i),image_patch[i])
		cv2.imshow('bs',plot_img)
		k = cv2.waitKey(0) 
		if k == 27:
			cv2.destroyAllWindows()
		print(image_patch.dtype)
		print(np.unique(image_patch[0]))
		self.model.single_out(image_patch)
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
		
img =cv2.imread('ab1.jpg',0)
_,img= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
Parser=parser_worm_image()
Parser.parser_image(img)


