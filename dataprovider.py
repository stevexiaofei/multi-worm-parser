import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
from random import randint
from utils import process_config
cfg= process_config('config.cfg')
# path= "D:\\dataset\\deepworm\\BBBC010_v1_foreground_eachworm\\BBBC010_v1_foreground_eachworm"
# files =os.listdir(path)
# f_name = lambda f:os.path.join(path,f)
# files=files[1:]
# contours=[]
# rects=[]

# for i,it in enumerate(files):
    # img=cv2.imread(f_name(it),0)
    # (_,cnts, hier) = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # if len(cnts)==1:
        # (x, y, w, h) = cv2.boundingRect(cnts[0])
        # contours.append(np.squeeze(cnts[0], axis=1))
        # rects.append((x, y, w, h,x+w/2,y+h/2))
# rects = np.array(rects)
# np.savez('worm_data.npz',cnts = contours,rects = rects)
class data_provider():
	def __init__(self,cfg):
		data_dict = np.load(cfg['dataset_path'])
		self.cfg = cfg
		self.contours=data_dict['cnts']
		self.rects=data_dict['rects']
		self.num=len(self.rects)
		self.num_samples = 1024
		self.center_samples=[]
		np.random.seed(0)
		for i in range(self.num_samples):
			center_index= np.random.randint(0,self.num-1)
			random_angle = np.random.random()*360
			cnt = self.transfer_loc(self.contours[center_index],self.rects[center_index],random_angle,scale=1.0).astype(np.int32)
			p= self.generate_inner_points(cnt)
			cnt=cnt-p
			cnt+=np.array([128,128])
			self.center_samples.append(cnt)	
	def __call__(self, num_sample):
		input_image_batch = np.zeros((num_sample,256,256,1),dtype=np.float32)
		output_image_batch = np.zeros((num_sample,256,256,1),dtype=np.float32)
		weight_mask = np.zeros((num_sample,256,256,1),dtype=np.float32)
		midout_image_batch = np.zeros((num_sample,64,64,1),dtype=np.float32)
		label_batch = np.zeros((num_sample,self.num_samples),dtype=np.int32)
		for i in range(num_sample):
			in_img,out_img,w_mask,mid_out,label= self.generater_pairwise_sample()

			input_image_batch[i]= in_img[...,np.newaxis]
			output_image_batch[i]=out_img[...,np.newaxis]
			midout_image_batch[i]=mid_out[...,np.newaxis]
			weight_mask[i] = w_mask[...,np.newaxis]
			label_batch[i][label]= 1
		input_image_batch -= np.amin(input_image_batch)
		input_image_batch /= np.amax(input_image_batch)
		output_image_batch -= np.amin(output_image_batch)
		output_image_batch /= np.amax(output_image_batch)
		midout_image_batch -= np.amin(midout_image_batch)
		midout_image_batch /= np.amax(midout_image_batch)
		return input_image_batch, output_image_batch,weight_mask,midout_image_batch, label_batch
	def transfer_loc(self,contour,rects,angle,scale=1.0,center=None):
		x,y,w,h,cx,cy = rects
		center =(cx,cy) if center is None else center
		angle =angle/180.*np.pi
		contour = contour-np.array([cx,cy])
		rotate = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
		contour = np.dot(rotate,contour.T).T*scale
		contour +=np.array(center)
        #todo maybe add a filter to filter the points that are out of bound
		return contour
	def generate_inner_points(self,cnt):
		(x,y,w,h)=cv2.boundingRect(cnt)
		x_list=(x+w*np.random.random((10,))).astype(np.int)
		cnt_points=filter(lambda x: x[0]>x_list[0]-2 and x[0]<x_list[0]+2,cnt)
		cnt_points=list(cnt_points)
		y_loc_ =[p[1] for p in cnt_points]
		y_max=max(y_loc_)
		y_min=min(y_loc_)
		h=y_max-y_min
		y_list=(y_min+h*np.random.random((10,))).astype(np.int)
		x_list=np.full(x_list.shape,x_list[0])
        #print('x_list',x_list)
		points=np.array([x_list,y_list]).T
        #print(x_list[0])
        #print(points)
        #print((x,y,w,h))
        #res =[cv2.pointPolygonTest(cnt,tuple(point),False) for point in points]
		res_ =[cv2.pointPolygonTest(cnt,tuple(point),True) for point in points]
		index =res_.index(max(res_))
        #print('index',index)
		return points[index]
	def add_other_worms(self,mask,cnt,rect,center_point):
		x,y,w,h,cx,cy = rect
		cnt=cnt-np.array([cx,cy],dtype=np.int32)
		center=np.random.random((2,))*256
		cnt_=cnt+np.array([center[0],center[1]],dtype=np.int32)

		while cv2.pointPolygonTest(cnt_,center_point,False)==1:
			center=(np.random.random((2,))*256).astype(np.uint32)
			cnt_=cnt+np.array([center[0],center[1]])
		cv2.drawContours(mask, [cnt_], -1, 255, thickness=-1)
        #mask=cv2.cv2.polylines(mask,[cnt_],True,[255,0,0],1)
		return mask
	def generater_pairwise_sample(self):
		label=np.random.randint(0,self.num_samples)
		if self.cfg['use_aux_cls']:
			mask_=cv2.drawContours(np.zeros((256,256),dtype=np.uint8),\
						[self.center_samples[label]], -1,255, thickness=-1)
		else:
			center_index = np.random.randint(0,self.num)
			random_angle = np.random.random()*360
			cnt = self.transfer_loc(self.contours[center_index],self.rects[center_index],random_angle,scale=1.0).astype(np.int32)
			p = self.generate_inner_points(cnt)
			cnt=cnt-p
			cnt+=np.array([128,128])
			mask_=cv2.drawContours(np.zeros((256,256),dtype=np.uint8),\
						[cnt], -1,255, thickness=-1)
		mask = cv2.dilate(mask_,np.ones((3,3)))-cv2.erode(mask_,np.ones((3,3)))
		mask[mask!=0]=4
		mask+=1
		singleout= mask_.copy()
		mid_out = cv2.resize(mask_,(64,64))
		for i in range(4):
			random_index=np.random.randint(0,self.num-1)
			mask_=self.add_other_worms(mask_, self.contours[random_index], self.rects[random_index], (128,128))
		return mask_, singleout, mask.astype(np.int32), mid_out, label
	def generator_test_sample(self):
		center_index = np.random.randint(0,self.num)
		random_angle = np.random.random()*360
		cnt = self.transfer_loc(self.contours[center_index],self.rects[center_index],random_angle,scale=1.0).astype(np.int32)
		p = self.generate_inner_points(cnt)
		cnt=cnt-p
		cnt+=np.array([128,128])
		mask_=cv2.drawContours(np.zeros((256,256),dtype=np.uint8),\
              		[cnt], -1,255, thickness=-1)
		singleout= mask_.copy()
		for i in range(4):
			random_index=np.random.randint(0,self.num-1)
			mask_=self.add_other_worms(mask_, self.contours[random_index], self.rects[random_index], (128,128))
		return mask_, singleout
	def generator_train(self,num_batch,batch_size=4):
		for i in range(num_batch):
			input_image_batch = np.zeros((batch_size,256,256,1),dtype=np.float32)
			output_image_batch = np.zeros((batch_size,256,256,1),dtype=np.float32)
			weight_mask = np.zeros((batch_size,256,256,1),dtype=np.float32)
			midout_image_batch = np.zeros((batch_size,64,64,1),dtype=np.float32)
			label_batch = np.zeros((batch_size,self.num_samples),dtype=np.int32)
			for i in range(batch_size):
				in_img,out_img,w_mask,mid_out,label= self.generater_pairwise_sample()
				input_image_batch[i]= in_img[...,np.newaxis]
				output_image_batch[i]=out_img[...,np.newaxis]
				midout_image_batch[i]=mid_out[...,np.newaxis]
				weight_mask[i] = w_mask[...,np.newaxis]
				label_batch[i][label]= 1
			input_image_batch -= np.amin(input_image_batch)
			input_image_batch /= np.amax(input_image_batch)
			output_image_batch -= np.amin(output_image_batch)
			output_image_batch /= np.amax(output_image_batch)
			midout_image_batch -= np.amin(midout_image_batch)
			midout_image_batch /= np.amax(midout_image_batch)
			yield input_image_batch, output_image_batch,weight_mask,midout_image_batch, label_batch		
	def generator_test(self,num_batch,batch_size=4):
		for i in range(num_batch):
			input_image_batch = np.zeros((batch_size,256,256,1),dtype=np.float32)
			output_image_batch = np.zeros((batch_size,256,256,1),dtype=np.float32)
			for i in range(batch_size):
				in_img,out_img = self.generator_test_sample()
				input_image_batch[i]= in_img[...,np.newaxis]
				output_image_batch[i]=out_img[...,np.newaxis]
			input_image_batch -= np.amin(input_image_batch)
			input_image_batch /= np.amax(input_image_batch)
			output_image_batch -= np.amin(output_image_batch)
			output_image_batch /= np.amax(output_image_batch)
			yield input_image_batch, output_image_batch

if __name__=='__main__':
	gene = data_provider(cfg)
	mask_, singleout, mask, mid_out, label = gene.generater_pairwise_sample()
	input_image_batch, output_image_batch, weight_mask,midout_image_batch, label_batch =gene(2)
	print(input_image_batch.dtype)
	print(input_image_batch.shape)
	img1=np.concatenate([input_image_batch[0,:,:,0],output_image_batch[0,:,:,0]],axis=1)
	img2=np.concatenate([input_image_batch[1,:,:,0],output_image_batch[1,:,:,0]],axis=1)
	img_ =np.concatenate([img1,img2],axis=0)
	#scipy.misc.imsave('img.jpg', np.concatenate([img,label],axis=1))
	print(img_.shape,img_.dtype)
	print(label_batch)
	scipy.misc.imsave('ab.jpg', img_)