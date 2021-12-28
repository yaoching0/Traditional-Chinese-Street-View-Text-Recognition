import torch
import pandas as pd
from PIL import ImageDraw,ImageFont,Image
import os
import numpy as np
import timm
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.data.parsers.parser_image_folder import find_images_and_targets
from timm.data import create_dataset, create_loader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#生成用於模型融合的訓練資料
#format:
#       [[5122x3,label(class id)],
#        [        ......        ],
#        [        ......        ]]
def model_inference(img_list,model1,model2,model3,idx_to_class,transform1,transform2,transform3):
    #直接回傳該中文資源的字串,如:'醫'
    #model1:effv2 model2:nfnet model3:vit
    #獲得分別用兩個模型對應的transform轉換得到的img1 img2
    stack_list1=[]
    stack_list2=[]
    stack_list3=[]
    for img in img_list:
        img1=transform1(img.copy())
        img1=img1.type(torch.FloatTensor)
        stack_list1.append(img1)

        img2=transform2(img.copy())
        img2=img2.type(torch.FloatTensor)
        stack_list2.append(img2)

        img3=transform3(img.copy())
        img3=img3.type(torch.FloatTensor)
        stack_list3.append(img3)

    img1=torch.stack(stack_list1,dim=0)
    img2=torch.stack(stack_list2,dim=0)
    img3=torch.stack(stack_list3,dim=0)


    #分別用兩個模型得到預測結果
    predict_result1=model1(img1.cuda())
    predict_result2=model2(img2.cuda())
    predict_result3=model3(img3.cuda())

    #定義softmax
    sm=torch.nn.Softmax(dim=1)

    predict_result1=sm(predict_result1)
    predict_result2=sm(predict_result2)
    predict_result3=sm(predict_result3)

    #按照0維進行拼接
    predict_cat=torch.cat((predict_result1,predict_result2,predict_result3),1)
    
    return predict_cat


#加載effV2模型的對照表
#-----------修改類別數這裡要改------------
samples, class_to_idx=find_images_and_targets(folder=r'C:\Users\yaoching\Desktop\繁中場景\chinese_classification_dataset_5122\validation',class_to_idx=None)
#samples, class_to_idx=find_images_and_targets(folder=r'C:\Users\yaoching\Desktop\繁中場景\chinese_classification_dataset_5049\validation',class_to_idx=None)

#鍵值互換
idx_to_class={}
for key,value in class_to_idx.items():
    idx_to_class[value] = key

#加載effv2_model模型,共5122類中文字
#effv2_model= timm.create_model('efficientnetv2_s', num_classes=5049,checkpoint_path=r'C:\Users\yaoching\Desktop\繁中場景\checkpoint-69_5049.pth.tar')
effv2_model= timm.create_model('efficientnetv2_s', num_classes=5122,checkpoint_path=r'C:\Users\yaoching\Desktop\繁中場景\effv2-rotate-epoch-82(98.09%).pth.tar')
data_config_effv2=effv2_model.default_cfg
effv2_model.eval()
effv2_model.cuda()

#加載nfnet模型
nfnet_model= timm.create_model('eca_nfnet_l2', num_classes=5122,checkpoint_path=r'C:\Users\yaoching\Desktop\繁中場景\new-nfnet-checkpoint-45(97.61%).pth.tar')
data_config_nfnet=nfnet_model.default_cfg
nfnet_model.eval()
nfnet_model.cuda()

#加載VIT模型
vit_model= timm.create_model('vit_base_patch16_384', num_classes=5122,checkpoint_path=r'C:\Users\yaoching\Desktop\繁中場景\new-vit-checkpoint-55(97.88%).pth.tar')
data_config_vit=vit_model.default_cfg
vit_model.eval()
vit_model.cuda()

#加載effV2 inference時需要的transfom
effv2_transform=transforms_imagenet_eval(
    img_size=data_config_effv2['input_size'][-2:],
    interpolation=data_config_effv2['interpolation'],
    mean=data_config_effv2['mean'],
    std=data_config_effv2['std'],
    crop_pct=data_config_effv2['crop_pct'],)

#加載nfnet inference時需要的transfom
nfnet_transform=transforms_imagenet_eval(
    img_size=data_config_nfnet['input_size'][-2:],
    interpolation=data_config_nfnet['interpolation'],  
    mean=data_config_nfnet['mean'],
    std=data_config_nfnet['std'],
    crop_pct=data_config_nfnet['crop_pct'],)

#加載vit inference時需要的transfom
vit_transform=transforms_imagenet_eval(
    img_size=data_config_vit['input_size'][-2:],
    interpolation=data_config_vit['interpolation'],
    mean=data_config_vit['mean'],
    std=data_config_vit['std'],
    crop_pct=data_config_vit['crop_pct'],)
count=0
input_path=r'C:\Users\yaoching\Desktop\繁中場景\chinese_classification_dataset_5122\validation'
output_path=r'C:\Users\yaoching\Desktop'
batch_size=2

feature=pd.DataFrame()
image_list=[]
#----遍歷test_path下每一張照片----
for filename in tqdm(os.listdir(input_path)):

    for c in os.listdir(os.path.join(input_path,filename)):
        image=Image.open(os.path.join(input_path,filename,c))
        image_list.append(image)

        #batch_size=2
        if(len(image_list)==batch_size or c==os.listdir(os.path.join(input_path,filename))[-1]):
            predict=model_inference(image_list,effv2_model,nfnet_model,vit_model,idx_to_class,effv2_transform,nfnet_transform,vit_transform)
            #print(len(image_list))
            image_list=[]
            class_id=class_to_idx[filename]

            #如果是1維的list，是按列添加的，二維才是按照行添加
            #https://blog.csdn.net/sinat_29957455/article/details/84961936
            predict=predict.cpu().detach().numpy().tolist()
            
            #list中的每一行都加入label
            for p in range(len(predict)):
                predict[p]=predict[p]+[class_id]

            #predict [batch_size,5122x3+1]:[[b1:5122x3+1],[b2:5122x3+1],...]
            feature=feature.append(predict)
            
print('DONE')
#df[range(10)].to_csv(os.path.join(test_path,'submission.csv'),header=False,index=False,encoding="utf_8_sig")
feature.to_csv(os.path.join(output_path,'ensemble_feature.csv'),index=False,header=False,encoding="utf_8_sig")
    #break
