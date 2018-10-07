import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
import sys
from random import randint
path= "D:\\dataset\\deepworm\\BBBC010_v1_foreground_eachworm\\BBBC010_v1_foreground_eachworm"
files =os.listdir(path)
f_name = lambda f:os.path.join(path,f)
files=files[1:]
contours=[]
rects=[]

for i,it in enumerate(files):
    img=cv2.imread(f_name(it),0)
    (_,cnts, hier) = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(cnts)==1:
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        contours.append(np.squeeze(cnts[0], axis=1))
        rects.append((x, y, w, h,x+w/2,y+h/2))
rects = np.array(rects)
Color_list=[(220,20,60),(255,0,255),(138,43,226),(0,0,255),(240,248,255),
(0,255,255),(0,255,127),(0,255,0),(255,255,0),(255,165,0),
(255,69,0),(128,0,0),(255,255,255),(188,143,143)]
Color_name=['Crimson','Magenta','BlueViolet','Blue','AliceBlue',
'Cyan','MediumSpringGreen','Lime','Yellow','Orange',
'OrangeRed','Maroon','White','RosyBrown']
class data_provider():
    def __init__(self,contours,rects):
        
        self.contours=contours
        self.rects=rects
        self.num=len(rects)
    def __call__(self, num_sample):
       batch = np.zeros((num_sample,516,516,2))
       for i in range(num_sample):
           img, label= self.generater_pairwise_sample()
           realAB=np.stack([img,label],axis=2)
           batch[i,:,:,:]=realAB
       batch -= np.amin(batch)
       batch /= np.amax(batch)
       return batch
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
    def add_other_worms(self,mask,cnt,rect,x_lim=(0,256),y_lim=(0,256),center_point=None,Parser_Worms=None,color=(255,255,255)):
	     
        x,y,w,h,cx,cy = rect
        x_min,x_max = x_lim
        y_min,y_max = y_lim
        center = (randint(x_min,x_max-1),randint(y_min,y_max-1))
        cnt=cnt-np.array([cx,cy],dtype=np.int32)
        cnt_=cnt+np.array([center[0],center[1]],dtype=np.int32)
        while center_point and  cv2.pointPolygonTest(cnt_,center_point,False)==1:
            center=(np.random.random((2,))*256).astype(np.uint32)
            cnt_=cnt+np.array([center[0],center[1]])
        cv2.drawContours(mask, [cnt_], -1, (255,255,255), thickness=-1)
        if Parser_Worms is not None:
            cv2.drawContours(Parser_Worms, [cnt_], -1, color, thickness=1)
    def generater_pairwise_sample(self):
        center_index= randint(0,self.num-1)
        random_angle = np.random.random()*360
        cnt = self.transfer_loc(contours[center_index],rects[center_index],random_angle,scale=1.0).astype(np.int32)
        p= self.generate_inner_points(cnt)
        cnt=cnt-p
        cnt+=np.array([128,128])
        mask_=cv2.drawContours(np.zeros((256,256,3),dtype=np.uint8), [cnt], -1,(255,255,255), thickness=-1)
        label= mask_.copy()
        for i in range(4):
            random_index= randint(0,self.num-1)
            mask_=self.add_other_worms(mask_, self.contours[random_index], rects[random_index],center_point=(128,128))
        return mask_, label
    def generater_sample(self):
        Panel= np.zeros((512,512,3),dtype=np.uint8)
        Parser_Worms= np.zeros((512,512,3),dtype=np.uint8)
        for i in range(20):
            random_index= randint(0,self.num-1)
            self.add_other_worms(Panel,self.contours[random_index],self.rects[random_index],\
													x_lim=(50,462),y_lim=(50,462),Parser_Worms=Parser_Worms,color=Color_list[i%len(Color_list)])
        return Panel,Parser_Worms
gene = data_provider(contours,rects)
if __name__=='__main__':
	gene = data_provider(contours,rects)
	img,Parser_Worms= gene.generater_sample()
	#sys.stdout.flush()
	scipy.misc.imsave('ab.jpg', img)
	scipy.misc.imsave('Parser_Worms.jpg', Parser_Worms)