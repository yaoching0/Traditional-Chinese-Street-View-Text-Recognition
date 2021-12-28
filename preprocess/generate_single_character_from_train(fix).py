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

#从比赛的train data 的json切出effv2辨识中文字元的crop 組成dataset
#12/15 注意原程式有bug,切出來的資料會被部分覆蓋掉
path=r'C:\Users\yaoching\Desktop\tradtional-chinese\train\json'
output_path=r'C:\Users\yaoching\Desktop\test_ds'

#draw=ImageDraw.ImageDraw(image)
for filename in tqdm(os.listdir(path)):
    #讀對應json檔的img
    image = Image.open(os.path.join(r'C:\Users\yaoching\Desktop\tradtional-chinese\train\img',filename.split('.')[0]+'.jpg'))
    with open(os.path.join(path,filename),'r',encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
        #print(load_dict['shapes'])

    for i in range(len(load_dict['shapes'])):
        #groupid為1或4
        if(load_dict['shapes'][i]['group_id']==1 or load_dict['shapes'][i]['group_id']==4):   
            points=load_dict['shapes'][i]['points']
            #print(points)
            #尋找四邊形的：
            #最左 越小越好 todo:後來發現不確定第一個點就是左上角？可能需要重切重train一次分類模型
            #update-------------
            left=min(points[0][0],points[1][0],points[2][0],points[3][0])

            #最上 越小越好
            upper=min(points[0][1],points[1][1],points[2][1],points[3][1])

            #最右 越大越好     
            right=max(points[0][0],points[1][0],points[2][0],points[3][0])

            #最下 越大越好
            lower=max(points[0][1],points[1][1],points[2][1],points[3][1])

            crop=image.crop((left,upper,right,lower))

            #crop的字要存放的資料夾
            crop_path=os.path.join(output_path,load_dict['shapes'][i]['label'])
            if not os.path.exists(crop_path):
            # 如果不存在则创建目录
            # 创建目录操作函数
                os.makedirs(crop_path) 
            crop.save(os.path.join(crop_path,filename.split('.')[0]+'_'+str(i)+'.jpg'))
    #break

            
#image.show()
