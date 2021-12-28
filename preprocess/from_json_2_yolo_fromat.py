import timm
from pprint import pprint
import torch 
import cv2
from PIL import ImageDraw,ImageFont,Image
import numpy as np 
import os
import copy
import shutil
import csv
import pandas as pd
import json

import torchvision.transforms as transforms
import random
import numpy
import math
import random
import torch
from torchvision import transforms
import numpy
import os

#从比赛的json生成yolo的label(針對中英數字串)
path=r'C:\Users\yaoching\Desktop\tradtional-chinese\train\json'
output_path=r'C:\Users\yaoching\Desktop\tradtional-chinese\train\yolo-string-non-chinese'
point_path=r'C:\Users\yaoching\Desktop\tradtional-chinese\train\four_points'

for filename in os.listdir(path):
    with open(os.path.join(path,filename),'r',encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
 
    f=open(os.path.join(output_path,filename.split('.')[0]+'.txt'),"w")
    
    
    #保存四個點的坐標
    f_p=open(os.path.join(point_path,filename.split('.')[0]+'.txt'),"w")


    for i in range(len(load_dict['shapes'])):
        #groupid為1或4
        #if(load_dict['shapes'][i]['group_id'] in [0,2,3,4,5]):   
        if(load_dict['shapes'][i]['group_id'] in [2,3]):  
            points=load_dict['shapes'][i]['points']
            #print(points)
            left=float(min(points[0][0],points[1][0],points[2][0],points[3][0]))

            #最上 越小越好
            upper=float(min(points[0][1],points[1][1],points[2][1],points[3][1]))

            #最右 越大越好     
            right=float(max(points[0][0],points[1][0],points[2][0],points[3][0]))

            #最下 越大越好
            lower=float(max(points[0][1],points[1][1],points[2][1],points[3][1]))
            #print(left,upper,right,lower)

            #計算yolo label format
            x_center=(left+right)/2
            y_center=(upper+lower)/2
            width=abs(left-right)
            height=abs(upper-lower)
            
            f.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(x_center/float(load_dict['imageWidth']),
                                    y_center/float(load_dict['imageHeight']),
                                    width/float(load_dict['imageWidth']),
                                    height/float(load_dict['imageHeight'])))
            f_p.write("0 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(points[0][0],points[0][1],points[1][0],points[1][1],points[2][0],points[2][1],points[3][0],points[3][1]))
            
            
#image.show()
