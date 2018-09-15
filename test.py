import cv2
import os
import numpy  as np
path= "D:\dataset\deepworm\BBBC010_v1_foreground_eachworm\BBBC010_v1_foreground_eachworm"
files =os.listdir(path)
f_name = lambda f:os.path.join(path,f)
files=files[1:]
img=cv2.imread(f_name(files[1]),0)
(_,cnts, hier) = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
print(cnts[0].dtype,cnts[0].shape)
dist = cv2.pointPolygonTest(cnts[0],(50,50),True)
print(dist)