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
from tqdm import tqdm

#將RCTW轉為字串yolo label format
label_path=r'D:\RCTW\train_gts'
output_path=r'C:\Users\yaoching\Desktop\繁中-高級\RCTW_string_label'
img_path=r'D:\RCTW\train_images'
for filename in tqdm(os.listdir(label_path)):

    img=Image.open(os.path.join(img_path,filename.split('.')[0]+'.jpg'))

    f2=open(os.path.join(output_path,filename.split('.')[0]+'.txt'),"w")

    with open(os.path.join(label_path,filename), "r", encoding='utf-8-sig') as f:
        labels = f.readlines() #['output\\img_0028735.jpg,鼬\n',...]
        #對該照片的每個字串
        for label in labels:
            label=label.strip()
            label=label.split(',') #['0', '0.x', '0.x', '0.x', '0.x'](x,y,width,height)
            
            points=[]
            try:
                points.append([float(label[0]),float(label[1])])
                points.append([float(label[2]),float(label[3])])
                points.append([float(label[4]),float(label[5])])
                points.append([float(label[6]),float(label[7])])
            except:
                print(filename)
                assert False
            left=float(min(points[0][0],points[1][0],points[2][0],points[3][0]))

            #最上 越小越好
            upper=float(min(points[0][1],points[1][1],points[2][1],points[3][1]))

            #最右 越大越好     
            right=float(max(points[0][0],points[1][0],points[2][0],points[3][0]))

            #最下 越大越好
            lower=float(max(points[0][1],points[1][1],points[2][1],points[3][1]))
      
            #計算yolo label format
            x_center=(left+right)/2
            y_center=(upper+lower)/2
            width=abs(left-right)
            height=abs(upper-lower)
            
            if(x_center*y_center*width*height)<0:
                print(filename)
            
            #存成yolo format
            f2.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(x_center/img.size[0],
                                    y_center/img.size[1],
                                    width/img.size[0],
                                    height/img.size[1]))
  