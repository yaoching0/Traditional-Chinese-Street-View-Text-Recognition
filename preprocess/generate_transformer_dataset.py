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
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import numpy
import math
import random
import torch
from torchvision import transforms
import numpy
import os



#從比賽中給data生成transfomerLM用的資料集
path=r'C:\Users\yaoching\Desktop\tradtional-chinese\train\json'
tokens_path=r'C:\Users\yaoching\Desktop\transformer\tokens.txt'
output_path=r'C:\Users\yaoching\Desktop\transformer'

#將一些合併為一類的字符替換起來,如C和c
def replace_c(s):
    s=s.replace('c','C').replace('m','M').replace('o','O').replace('p','P').replace('s','S').replace('v','V').replace('w','W').replace('x','X').replace('z','Z')
    return s
#tokens
tokens=open(tokens_path, "r",encoding="utf-8")

#token列表,順便去除\n
token_list=tokens.readlines()
for i in range(len(token_list)):
    token_list[i]=token_list[i].strip()

output=pd.DataFrame(columns=['text'])

for filename in tqdm(os.listdir(path)):

    with open(os.path.join(path,filename),'r',encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    
    for i in range(len(load_dict['shapes'])):
        #挑選字串  
        temp_str=''
        if(load_dict['shapes'][i]['group_id'] in [0,2,3]):  
            
            #將一些在之後分類器已經替換掉的大小寫一致的也替換掉保持一致
            label=replace_c(load_dict['shapes'][i]['label'])
            for c in label:
                #如果不在token list則刪除掉
                if c in token_list:
                    temp_str+=c
            
            #如果長度大於等於2的話
            if len(temp_str)>1:
                output=output.append({'text':temp_str},ignore_index=True)

output.to_csv(os.path.join(output_path,'tfer-dataset.csv'),index=False,header=True,encoding="utf_8")

            

