import json
import pprint
import os
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
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.data.parsers.parser_image_folder import find_images_and_targets
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
from torch import nn
import time

#將CTW的每個sentence轉換成yolo label存至output
#兩個路徑要執行兩次
train_anno=r'C:\Users\yaoching\Desktop\繁中場景\ctw-annotations.tar\ctw-annotations\train.jsonl'
val_anno=r'C:\Users\yaoching\Desktop\繁中場景\ctw-annotations.tar\ctw-annotations\val.jsonl'

output_path=r'C:\Users\yaoching\Desktop\繁中-高級\ctw_string_label'
#每個element都是一張照片的標註(dict格式)
anno_list=[]

with open(val_anno) as f:
    temp_list=f.readlines()
    for i in temp_list:
        anno_list.append(json.loads(i))
        

#對每張照片
for label in tqdm(anno_list):
    #image=Image.open(os.path.join(r'C:\Users\yaoching\Desktop\繁中-高級\ctw',label['file_name']))
    #draw=ImageDraw.ImageDraw(image)

    #保存yolo label的路徑
    f_yolo=open(os.path.join(output_path,label['image_id']+'.txt'),"w")

    #該圖片中的每個sentence
    for s in range(len(label['annotations'])):
        xlist=[]
        ylist=[]
        #sentence中的每個instance
        for instance in label['annotations'][s]:
            points=instance['polygon']
            for c in range(4):
                xlist.append(points[c][0])
                ylist.append(points[c][1])
        x_np=np.array(xlist)
        y_np=np.array(ylist)

        #計算該句的左上角和右下角
        x_max=np.max(x_np)
        x_min=np.min(x_np)
        y_max=np.max(y_np)
        y_min=np.min(y_np)
        
        #計算該string yolo label format
        x_center=(x_max+x_min)/2
        y_center=(y_max+y_min)/2
        width=x_max-x_min
        height=y_max-y_min

        #確保不會出現負數
        if(x_center*y_center*width*height<0):
            continue
        #存成yolo label
        f_yolo.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(x_center/2048.,y_center/2048.,width/2048.,height/2048.))
        #draw.rectangle(((x_min, y_min),(x_max, y_max)), fill=None, outline='red', width=2) 
    
   
    
    '''
    #對該照片的每句話
    for s in label['annotations']:
        
        #對每句話中的每個字
        for c in s:
            if(c['is_chinese']==True):
                c_xcenter=c['adjusted_bbox'][0]+c['adjusted_bbox'][2]/2.
                c_ycenter=c['adjusted_bbox'][1]+c['adjusted_bbox'][3]/2.
                c_w=c['adjusted_bbox'][2]
                c_h=c['adjusted_bbox'][3]
                f_yolo.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(c_xcenter/2048.,c_ycenter/2048.,c_w/2048.,c_h/2048.))
        '''


r'''
#Ignore EDA
for g in anno_list[0]['ignore']:
    g_xcenter=g['bbox'][0]+g['bbox'][2]/2.
    g_ycenter=g['bbox'][1]+g['bbox'][3]/2.
    g_w=g['bbox'][2]
    g_h=g['bbox'][3]
    f_yolo.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(g_xcenter/2048.,g_ycenter/2048.,g_w/2048.,g_h/2048.))
    '''
#pprint.pprint(anno, depth=3)