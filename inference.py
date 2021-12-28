import torch
import pandas as pd
import argparse
from PIL import ImageDraw,ImageFont,Image
import os
import numpy as np
import timm
from data.transforms_factory import transforms_imagenet_eval
from timm.data import create_dataset, create_loader
from tqdm import tqdm
import warnings
from transformers import BertTokenizer,DistilBertForMaskedLM,DistilBertConfig
from torch import nn
import json
from EnsembleNet.EnsembleNet import EnsembleNet
from functions.functions import *
warnings.filterwarnings("ignore")

#判斷是(還是C
def process_C(sentence,classifier_conf):
    
    #統計字串中數字和英文字的數量
    count_num=0
    count_english=0

    #對reg_sentence中的每個字母做統計
    for c in sentence:
        #防止預測到旋轉類，如：l_0
        c=c.split('_')[-1]
        
        #要先encode()才對中文有效,不然中文會被統計為英文
        if c.encode().isalpha():
            count_english+=1
        elif c.isdigit():
            count_num+=1

    #數字比英文多,或英文數字一樣多，維持原狀
    if count_num>=count_english:
        for i in range(len(sentence)):
            if sentence[i]=='<':
                sentence[i]=''

                #防止後續再把它換成別的
                classifier_conf[i]=[1.,0.,0.]
    
    #英文比數字多
    elif count_num<count_english:
        for i in range(len(sentence)):
            if sentence[i]=='<':
                sentence[i]='C'
    
    #print(''.join(sentence))
    
    return sentence,classifier_conf

#判斷是0還是O
def process_O0(sentence):
    
    #統計字串中數字和英文字的數量
    count_num=0
    count_english=0

    #對reg_sentence中的每個字母做統計
    for c in sentence:
        #防止預測到旋轉類，如：l_0
        c=c.split('_')[-1]
        
        #要先encode()才對中文有效,不然中文會被統計為英文
        if c.encode().isalpha():
            count_english+=1
        elif c.isdigit():
            count_num+=1

    #數字比英文多 O -> 0
    if count_num>count_english:
        for i in range(len(sentence)):
            if sentence[i]=='O':
                sentence[i]='0'

            if sentence[i]=='l_O':
                sentence[i]='l_0'
    #英文比數字多 0 -> O
    elif count_num<count_english:
        for i in range(len(sentence)):
            if sentence[i]=='0':
                sentence[i]='O'
            if sentence[i]=='l_0':
                sentence[i]='l_O'

    #數字英文一樣多,維持原狀
    else:
        pass
    #print(''.join(sentence))
    
    return sentence
'''
用來配合NLP模型使用的四元組
format:(target character,number,english,chinese).
如果match到character，由NLP模型決定是中文、英文還是數字
'''
#12/09:取消了 ('1','1','I',None) 和 ('2','2','Z',None)
confusion_pairs=[('0','0','O',None),('O','0','O',None), #for (0,O,None)
                 ('I','1','I',None),('l','1','l',None), #for (1,l/I,None)
                 ('Z','2','Z',None),('l_N','2','l_N',None),  #for (2,Z/l_N,None)
                 ('l_1','l_1','l_I','一'),('l_l','l_1','l_l','一'),('l_I','l_1','l_I','一'),('一','l_1','l_I','一'),  #for (l_1,l_l/l_I,一)
                 ('l_2','l_2','N',None),('l_Z','l_2','l_Z',None),('N','l_2','N',None), #for (l_2,l_Z/N,None)
                 ('l_0','l_0','l_O',None),('l_O','l_0','l_O',None),]  #for (l_0,l_O,None)
#需要判斷的character
key_list=[i[0] for i in confusion_pairs]

#對於完全無法區分的字母，使用NLP模型進行進一步判斷，如數字0/英文O
def sentence_process_confusion(sentence,confusion_pairs,key_list,tokenizer,transformer_model,transformer_LM_model,classifier_word,classifier_conf):
    #記錄那些字元存在於混淆三元組中
    confusion_idx=[]
    #先對0和O單獨處理一次再丟入transformer
    sentence=process_O0(sentence)
    #保存去掉l的句子(list)
    non_l_sentence=[]
    for c in range(len(sentence)):
        non_l_sentence.append(sentence[c].split('_')[-1])

    #對reg_sentence中的每個字母做統計
    for i in range(len(sentence)):
        if sentence[i] in key_list:
            #confusion_idx的每個element為 (ith word in sentence , ith element in key_list)
            confusion_idx.append((i,key_list.index(sentence[i])))

    #對每個在confusion的element
    for c in confusion_idx:
        temp_sentence=non_l_sentence.copy()
        #將相應的element替換成mask
        temp_sentence[c[0]]=tokenizer.mask_token

        #轉換為token
        temp_sentence=tokenizer(temp_sentence,return_tensors="pt", padding="max_length", max_length=512,is_split_into_words=True)

        #transformer_model沒有這個參數
        del temp_sentence['token_type_ids']

        #放入GPU計算
        temp_sentence={k: v.to(device) for k, v in temp_sentence.items()}

        #label 0:數字 1:英文 2:中文
        output=transformer_model(**temp_sentence).logits[0,c[0]+1].detach().cpu().numpy().tolist()
        
        #decode出來看看mask的句子的樣子
        #print(tokenizer.decode(temp_sentence['input_ids'][0].detach().cpu().numpy().tolist()))
      
        #獲得排序後的位置索引(從大到小)
        output=np.argsort(output)[::-1]

        #將sentence中相應的character替換成辨識出的pair相應的結果
        if confusion_pairs[c[1]][output[0]+1] is not None:
            sentence[c[0]]=confusion_pairs[c[1]][output[0]+1]
        elif confusion_pairs[c[1]][output[1]+1] is not None:
            sentence[c[0]]=confusion_pairs[c[1]][output[1]+1]
        else:
            sentence[c[0]]=confusion_pairs[c[1]][output[2]+1]

    #12/11:新功能，根據信心度來決定是不是confused，並按信心度排序結果尋找最佳輸出
    #-------------------------------------------------------------
    #12/18:暫時停用此stage
    '''
    #(修正)保存去掉l的句子(list),重新按照新的sentence建立一次
    non_l_sentence=[]
    for c in range(len(sentence)):
        non_l_sentence.append(sentence[c].split('_')[-1])

    #清空，根據信心度再判斷一次那些會是confusion
    confusion_idx=[]

    #判斷那些是confusion
    for i in range(len(classifier_conf)):

        temp_conf=classifier_conf[i]

        if temp_conf[0]<0.5 and (temp_conf[0]/temp_conf[1])<5:
            confusion_idx.append(i)
    
    #對每個在confusion的element
    for c in confusion_idx:
        temp_sentence=non_l_sentence.copy()
        #將相應的element替換成mask
        temp_sentence[c]=tokenizer.mask_token

        #轉換為token
        temp_sentence=tokenizer(temp_sentence,return_tensors="pt", padding="max_length", max_length=512,is_split_into_words=True)

        #transformer_model沒有這個參數
        del temp_sentence['token_type_ids']

        #放入GPU計算
        temp_sentence={k: v.to(device) for k, v in temp_sentence.items()}

        #label 0:數字 1:英文 2:中文
        output=transformer_model(**temp_sentence).logits[0,c+1].detach().cpu().numpy().tolist()
      
        #獲得排序後的位置索引(從大到小)
        output=np.argsort(output)[::-1][0]
        
        #針對三類分別處理
        if output==0:
            if non_l_sentence[c].isdigit():
                continue
            else:
                for w in classifier_word[c][1:3]:

                    #如果辨識是符號則捨棄
                    if w.split('_')[0]=='inter':
                        continue

                    #小寫的類別名為重疊，如aa/bb，取單一字母
                    if w.isalpha() and w.islower():
                        w=w[0]

                    #l_ee -> l_e
                    elif ('_' in w) and w.split('_')[-1].islower():
                        w='l_'+w.split('_')[-1][0]

                    w=w.split('_')[-1]

                    if w.isdigit():
                        sentence[c]=w
                        continue

        elif output==1:
            if non_l_sentence[c].encode().isalpha():
                continue
            else:
                for w in classifier_word[c][1:3]:

                     #如果辨識是符號則捨棄
                    if w.split('_')[0]=='inter':
                        continue

                    #小寫的類別名為重疊，如aa/bb，取單一字母
                    if w.isalpha() and w.islower():
                        w=w[0]

                    #l_ee -> l_e
                    elif ('_' in w) and w.split('_')[-1].islower():
                        w='l_'+w.split('_')[-1][0]

                    w=w.split('_')[-1]
                    
                    if w.encode().isalpha():
                        sentence[c]=w
                        continue
        elif output==2:
            if is_all_chinese(non_l_sentence[c]):
                continue
            else:
                for w in classifier_word[c][1:3]:

                     #如果辨識是符號則捨棄
                    if w.split('_')[0]=='inter':
                        continue

                    #小寫的類別名為重疊，如aa/bb，取單一字母
                    if w.isalpha() and w.islower():
                        w=w[0]

                    #l_ee -> l_e
                    elif ('_' in w) and w.split('_')[-1].islower():
                        w='l_'+w.split('_')[-1][0]

                    w=w.split('_')[-1]
                    
                    if is_all_chinese(w):
                        sentence[c]=w
                        continue
    '''
    #12/13:根據信心度來決定是不是confused，並使用語言模型判斷辨識結果的前三項中,和語言模型的前五項中是否有重疊的結果。
    #-------------------------------------------------------------

    #保存去掉l的句子(list),重新按照新的sentence建立一次
    non_l_sentence=[]
    for c in range(len(sentence)):
        non_l_sentence.append(sentence[c].split('_')[-1])

    #清空，根據信心度再判斷一次那些會是confusion
    confusion_idx=[]

    #判斷那些是confusion
    for i in range(len(classifier_conf)):

        temp_conf=classifier_conf[i]

        if temp_conf[0]<0.5 and (temp_conf[0]/temp_conf[1])<5:
            confusion_idx.append(i)

    #對每個在confusion的element
    for c in confusion_idx:
        temp_sentence=non_l_sentence.copy()
        #temp_sentence=['早','餐','店'] #for test
        #將相應的element替換成mask
        temp_sentence[c]=tokenizer.mask_token
        
        #轉換為token
        temp_sentence=tokenizer(temp_sentence,return_tensors="pt", padding="max_length", max_length=512,is_split_into_words=True)

        #transformer_model沒有這個參數
        del temp_sentence['token_type_ids']

        #放入GPU計算
        temp_sentence={k: v.to(device) for k, v in temp_sentence.items()}

        #label 0:數字 1:英文 2:中文
        output=transformer_LM_model(**temp_sentence).logits[0,c+1].detach().cpu().numpy().tolist()
      
        #獲得排序後的位置索引(從大到小,前五)
        output=np.argsort(output)[:-6:-1]

        #將第一個字符替換成現有的
        classifier_word[c][0]=sentence[c]

        temp_word=[]
        #將classifier_word中的word格式化
        for w in classifier_word[c]:

            #如果辨識是符號則捨棄
            if w.split('_')[0]=='inter':
                temp_word.append(None)
                continue

            #小寫的類別名為重疊，如aa/bb，取單一字母
            if w.isalpha() and w.islower():
                w=w[0]

            #l_ee -> l_e
            elif ('_' in w) and w.split('_')[-1].islower():
                w='l_'+w.split('_')[-1][0]

            w=w.split('_')[-1]
            temp_word.append(w)
        
        top_5=[]
        for i in output:
            top_5.append(tokenizer.decode(i))
        
        for word in temp_word:
            if word in top_5:
                sentence[c]=word
                break
            
    return sentence
    
def model_inference(img,model1,model2,model3,ensemble_net,idx_to_class,transform1,transform2,transform3):
    #直接回傳該中文資源的字串,如:'醫'
    #model1:effv2 
    #model2:VIT
    #model3:nfnet

    #獲得分別用兩個模型對應的transform轉換得到的img1 img2
    img1=transform1(img.copy())
    img1=img1.unsqueeze(0).type(torch.FloatTensor)

    img2=transform2(img.copy())
    img2=img2.unsqueeze(0).type(torch.FloatTensor)

    img3=transform3(img.copy())
    img3=img3.unsqueeze(0).type(torch.FloatTensor)

    #分別用兩個模型得到預測結果
    predict_result1=model1(img1.cuda())
    predict_result2=model2(img2.cuda())
    predict_result3=model3(img3.cuda())

    #定義softmax
    sm=torch.nn.Softmax(dim=0)

    predict_result1=sm(predict_result1[0])
    predict_result2=sm(predict_result2[0])
    predict_result3=sm(predict_result3[0])

    #簡單模型融合
    predict_result_simple=0.4*predict_result1+0.3*predict_result2+0.3*predict_result3
    
    #對辨識字的信心度
    sorted, indices = torch.sort(predict_result_simple,dim=0,descending=True)

    #獲得信心度前三高的結果 並存入conf和word
    conf=sorted[:5].detach().cpu().numpy().tolist()
    
    word=indices[:5].detach().cpu().numpy().tolist()
    temp=[]
    
    #從index轉換成字符
    for w in word:
        temp.append(idx_to_class[w])

    word=temp

    #apply Res-EnsembleNet
    predict_result=torch.cat((predict_result1,predict_result2,predict_result3),0)
    predict_result=ensemble_net(predict_result.unsqueeze(0).type(torch.FloatTensor).cuda())
    predict_result=predict_result[0]
    word[0]=idx_to_class[predict_result.argmax(dim=0).item()]

    #回傳前三可能的結果和相應之信心度
    return conf,word


#加載必要參數
parser = argparse.ArgumentParser()
# base argument
parser.add_argument('--yolo-string', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights','yolo_string.pt'),
                        help='weight path of yolo for string detection')
parser.add_argument('--yolo-character', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights','yolo_characterV5(epoch179).pt'),
                        help='weight path of yolo for character detection')
parser.add_argument('--cls-model-eff', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights','effV2-FIX-checkpoint-88.pth.tar'),
                        help='weight path of effV2')
parser.add_argument('--cls-model-vit', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights','VIT-checkpoint-73.pth.tar'),
                        help='weight path of vit') 
parser.add_argument('--cls-model-nf', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights','NF-NET-checkpoint-41.pth.tar'),
                        help='weight path of NF-Net')   
parser.add_argument('--class-to-idx-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data','class_to_idx.json'),
                        help='path of classificaion dataset')
parser.add_argument('--image-file-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),r'input_images'),
                        help='path of image file to inference')  
parser.add_argument('--output-path', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help='path of folder of saving inference result')  
parser.add_argument('--tokenizer-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data',r'tokens.txt'),
                        help='path of tokenizer for NLP Block') 
parser.add_argument('--transformer-model-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights',r'model_saved_for_3_classes_weight.bin'),
                        help='path of 3 classification transformer weight') 
parser.add_argument('--transformer-LM-model-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights',r'model_saved_normal_LM_weight.bin'),
                        help='path of masked language model weight') 
parser.add_argument('--transformer-model-config-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'config',r'model_saved_for_3_classes_config.json'),
                        help='path of config to creating 3 classification transformer') 
parser.add_argument('--transformer-LM-model-config-path', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'config',r'model_saved_normal_LM_config.json'),
                        help='path of config to creating masked language model ')  
parser.add_argument('--Ensemble-weight', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'weights',r'Res-EnsembleNet-weight.pth'),
                         help='weight path of Res-EnsembleNet')                                          
cf = parser.parse_args()

#-----用於預測的圖片路徑-----
test_path=cf.image_file_path
#test_path=r'C:\Users\yaoching\Desktop\test'

#輸出csv路徑
output_path=cf.output_path

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    
    #加載用於判斷英文、數字和中文的transformer
    tokenizer = BertTokenizer(vocab_file=cf.tokenizer_path,do_lower_case=False)

    #加載config並建立model
    BertConfig=DistilBertConfig.from_pretrained(cf.transformer_model_config_path)
    transformer_model = DistilBertForMaskedLM(BertConfig)

    #根据新的vocab改變模型尺寸
    transformer_model.resize_token_embeddings(len(tokenizer))

    #修改輸出類別數
    transformer_model.vocab_projector=nn.Linear(768,3,bias=True)

    #加載訓練好的三分類transformer模型
    transformer_model.load_state_dict(torch.load(cf.transformer_model_path))

    #請在加載完後再移至GPU，這BUG我DE了半小時淦
    transformer_model.to(device)
    transformer_model.eval()

    #加載config並建立model
    BertConfig=DistilBertConfig.from_pretrained(cf.transformer_LM_model_config_path)
    transformer_LM_model = DistilBertForMaskedLM(BertConfig)

    #根据新的vocab改變模型尺寸
    transformer_LM_model.resize_token_embeddings(len(tokenizer))

    #加載普通語言模型
    transformer_LM_model.load_state_dict(torch.load(cf.transformer_LM_model_path))

    #請在加載完後再移至GPU
    transformer_LM_model.to(device)
    transformer_LM_model.eval()

    #用於保存最後輸出的dataframe
    #df=pd.DataFrame(columns=['imgName','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y','p4_x''p4_y','predictText'])
    df=pd.DataFrame()

    #加載用於偵測string的yolo模型
    #add CTW re train weight
    yolo_string = torch.hub.load('ultralytics/yolov5', 'custom', path=cf.yolo_string)  # or yolov5m, yolov5l, yolov5x, custom
    yolo_string.eval()

    #設定yolo閥值
    yolo_string.conf=0.2 
    yolo_string.iou = 0.33 

    # 加載用於偵測character的yolo模型
    #add CTW re train weight
    yolo_character = torch.hub.load('ultralytics/yolov5', 'custom', path=cf.yolo_character)  # or yolov5m, yolov5l, yolov5x, custom
    yolo_character.eval()

    #設定yolo閥值
    yolo_character.conf=0.2 #0.03125
    yolo_character.iou = 0.33 #default 0.45


    #加載effV2模型的對照表
    with open(cf.class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    
    #鍵值互換
    idx_to_class={}
    for key,value in class_to_idx.items():
        idx_to_class[value] = key

    #加載模型融合網路
    ensemble_net=EnsembleNet()
    ensemble_net.load_state_dict(torch.load(cf.Ensemble_weight))
    ensemble_net.cuda()

    #加載effv2_model模型,共5073(12/08:5122 增加了向右旋轉90讀的數字和英文)類中文字
    effv2_model= timm.create_model('efficientnetv2_s', num_classes=5122,checkpoint_path=cf.cls_model_eff) #原61
    data_config_effv2=effv2_model.default_cfg
    effv2_model.cuda()
    effv2_model.eval()

    #加載VIT模型
    vit_model= timm.create_model('vit_base_patch16_384', num_classes=5122,checkpoint_path=cf.cls_model_vit)
    data_config_vit=vit_model.default_cfg
    vit_model.eval()
    vit_model.cuda()

    #加載nfnet模型
    nfnet_model= timm.create_model('eca_nfnet_l2', num_classes=5122,checkpoint_path=cf.cls_model_nf)
    data_config_nfnet=nfnet_model.default_cfg
    nfnet_model.eval()
    nfnet_model.cuda()

    #加載effV2 inference時需要的transfom
    effv2_transform=transforms_imagenet_eval(
        img_size=data_config_effv2['input_size'][-2:],
        interpolation=data_config_effv2['interpolation'],
        mean=data_config_effv2['mean'],
        std=data_config_effv2['std'],
        crop_pct=data_config_effv2['crop_pct'],)

    #加載vit inference時需要的transfom
    vit_transform=transforms_imagenet_eval(
        img_size=data_config_vit['input_size'][-2:],
        interpolation=data_config_vit['interpolation'],
        mean=data_config_vit['mean'],
        std=data_config_vit['std'],
        crop_pct=data_config_vit['crop_pct'],)

    #加載nfnet inference時需要的transfom
    nfnet_transform=transforms_imagenet_eval(
        img_size=data_config_nfnet['input_size'][-2:],
        interpolation=data_config_nfnet['interpolation'],  
        mean=data_config_nfnet['mean'],
        std=data_config_nfnet['std'],
        crop_pct=data_config_nfnet['crop_pct'],)

    #----遍歷test_path下每一張照片----
    for filename in tqdm(os.listdir(test_path)):
        #image_df為該圖片的所有框之坐標
        image=Image.open(os.path.join(test_path,filename))

        #用yolo計算filename這張圖的所有string框並存在results_str_df
        results_str = yolo_string(image,size=1280)
        #results_str.show()
        results_str_df=results_str.pandas().xyxy[0]
        
        results_str_df['filename']=filename.split('.')[0]

        #計算yolo檢測出每個框的中心點
        results_str_df['x_center']=(results_str_df['xmin']+results_str_df['xmax'])/2
        results_str_df['y_center']=(results_str_df['ymin']+results_str_df['ymax'])/2

        #計算寬和高
        results_str_df['width']=(results_str_df['xmax']-results_str_df['xmin'])
        results_str_df['height']=(results_str_df['ymax']-results_str_df['ymin'])
        results_str_df['area']=results_str_df['width']*results_str_df['height']
        
        results_str_df['predictText']='###'

        #用來儲存輸出string的四個點坐標值(順時針)
        results_str_df['p1_x']=None
        results_str_df['p1_y']=None
        results_str_df['p2_x']=None
        results_str_df['p2_y']=None
        results_str_df['p3_x']=None
        results_str_df['p3_y']=None
        results_str_df['p4_x']=None
        results_str_df['p4_y']=None

        #用yolo計算filename這張圖的所有character框並存在results_character_df
        results_character = yolo_character(image,size=1280)
        #results_character.show()

        results_character_df=results_character.pandas().xyxy[0]
        
        #計算yolo檢測出每個框的中心點
        results_character_df['x_center']=(results_character_df['xmin']+results_character_df['xmax'])/2
        results_character_df['y_center']=(results_character_df['ymin']+results_character_df['ymax'])/2
        results_character_df['assign']='NULL'
        
        #results_character.show()
        #print(results_character_df)

        #遍歷result_character_df，並將其結果分配給results_str_df中最合適的框
        for ri in range(results_character_df.shape[0]):
            r_center=np.array([results_character_df['x_center'].iloc[ri],results_character_df['y_center'].iloc[ri]])
            area=1e9
            #對每個string框
            for s in range(results_str_df.shape[0]):
                s_area=results_str_df['area'].iloc[s]
                #將其分配給包含它的面積最小的sting框
                if area > s_area and is_in_bbx(results_str_df.iloc[s],r_center):
                    area=s_area
                    results_character_df['assign'].iloc[ri]=results_str_df.iloc[s].name

        #對該照片的每一個框遍歷
        for i in range(results_str_df.shape[0]):
            #記錄該框所有character的classifier前三高信心度的結果 format:[['D', 'O', 'l_O'],[.,.,.],...]
            classifier_word=[]

            #記錄該框所有character的classifier前三高信心度的結果 [[0.87, 0.11, 0.1],[.,.,.],...]
            classifier_conf=[]

            #被分配給這個string的character
            assign_df=results_character_df[results_character_df['assign']==results_str_df.loc[i].name]
            
            #確定該字串中是否包含(
            has_inter_2=False

            #如果這個string沒有被分配到任意一個字元就捨棄
            if(assign_df.shape[0]==0):
                results_str_df=results_str_df.drop([i])
                continue

            #string bbx的寬和高
            width=results_str_df.loc[i,'xmax']-results_str_df.loc[i,'xmin']
            height=results_str_df.loc[i,'ymax']-results_str_df.loc[i,'ymin']
            
            reg_sentence=[]
            
            #由左到右
            if(width>=height):

                #以x_center由小到大排序
                assign_df=assign_df.sort_values(by='x_center',ascending=True).reset_index(drop=True)
                
            #由上到下
            else:
                
                #以y_center由小到大排序
                assign_df=assign_df.sort_values(by='y_center',ascending=True).reset_index(drop=True)

            #以由左到右/上到下的順序遍歷assign_df的每個yolo檢測字元
            for j in range(assign_df.shape[0]):
                character_crop=image.crop((assign_df.loc[j,'xmin'],assign_df.loc[j,'ymin'],assign_df.loc[j,'xmax'],assign_df.loc[j,'ymax']))

                #回傳信心度和相應的字
                conf_3,word_3=model_inference(character_crop,effv2_model,vit_model,nfnet_model,ensemble_net,idx_to_class,effv2_transform,vit_transform,nfnet_transform)

                #默認取幾率最高的結果
                fusion_model_conf=conf_3[0]
                word=word_3[0]

                #綜合考慮辨識模型信心度及yolo信心度覺決定是否捨棄該框
                if (fusion_model_conf<0.1 and assign_df['confidence'][j]<0.6) or fusion_model_conf<0.01 or (fusion_model_conf+assign_df['confidence'][j])<0.68:
                    assign_df=assign_df.drop([j])
                    continue
                
                #如果辨識是符號則捨棄(,'('因為和C太像會特殊處理 
                if word.split('_')[0]=='inter':
                    if word.split('_')[1]!='2':
                        assign_df=assign_df.drop([j])
                        continue

                    #暫且先使其為<，後續再決定要捨棄它還是使其變為C
                    else:
                        has_inter_2=True
                        word='<'

                #小寫的類別名為重疊，如aa/bb，取單一字母
                if word.isalpha() and word.islower():
                    word=word[0]
                    
                #將l_ee -> l_e
                elif ('_' in word) and word.split('_')[-1].islower():
                    word='l_'+word.split('_')[-1][0]

                reg_sentence.append(word)

                #將信心度前三高的結果保存起來
                classifier_word.append(word_3)
                classifier_conf.append(conf_3)
    
            #如果這個string沒有被分配到任意一個字元就捨棄
            if(assign_df.shape[0]==0):
                results_str_df=results_str_df.drop([i])
                continue
            
            #最終輸出字串的bbx由被分配到的字串重新修正
            #12/17:修改為計算幾何區塊
            out_string_bbx=computational_geometry_block(assign_df)
            results_str_df.loc[i,'p1_x']=out_string_bbx[0][0]
            results_str_df.loc[i,'p1_y']=out_string_bbx[0][1]
            results_str_df.loc[i,'p2_x']=out_string_bbx[1][0]
            results_str_df.loc[i,'p2_y']=out_string_bbx[1][1]
            results_str_df.loc[i,'p3_x']=out_string_bbx[2][0]
            results_str_df.loc[i,'p3_y']=out_string_bbx[2][1]
            results_str_df.loc[i,'p4_x']=out_string_bbx[3][0]
            results_str_df.loc[i,'p4_y']=out_string_bbx[3][1]

        
            #如果存在O或0 或(，啟動判斷
            if  has_inter_2==True:
                
                #判斷是C還是(
                reg_sentence,classifier_conf=process_C(reg_sentence,classifier_conf)

                #防止歸零
                if len(reg_sentence)==0:
                    results_str_df=results_str_df.drop([i])
                    continue
            
            #使用NLP模型進一步處理
            if len(reg_sentence)>1:
                reg_sentence=sentence_process_confusion(reg_sentence,confusion_pairs,key_list,tokenizer,transformer_model,transformer_LM_model,classifier_word,classifier_conf)
            
            temp_sen=''
            #去掉l_
            for c in reg_sentence:
                temp_sen+=c.split('_')[-1]
            reg_sentence=temp_sen

            #NA提交時會報錯，使其维持在'###'
            if reg_sentence!='NA':
                results_str_df.loc[i,'predictText']=reg_sentence
                    

        results_str_df=results_str_df[['filename','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y','p4_x','p4_y','predictText']]

        #對於沒被分配到任何框但信心度高於0.5的character
        assign_null=results_character_df[(results_character_df['assign']=='NULL') & (results_character_df['confidence']>0.5)].reset_index(drop=True)

        #將其單獨作為一個string輸出
        if assign_null.shape[0]>0:
            assign_null['filename']=filename.split('.')[0]
            assign_null['predictText']='###'
            for p in range(assign_null.shape[0]):
                #這邊因為前面有reset index 才可以用 loc 而不是 iloc
                character_crop=image.crop((assign_null.loc[p,'xmin'],assign_null.loc[p,'ymin'],assign_null.loc[p,'xmax'],assign_null.loc[p,'ymax']))
                
                #回傳信心度和相應的字
                conf_3,word_3=model_inference(character_crop,effv2_model,vit_model,nfnet_model,ensemble_net,idx_to_class,effv2_transform,vit_transform,nfnet_transform)

                #默認取幾率最高的結果
                fusion_model_conf=conf_3[0]
                word=word_3[0]

                #如果yolo和classifier的信心度加總後超過1.0 則作為單獨的字串保存並輸出
                if (fusion_model_conf + assign_null.loc[p,'confidence'])>1.:

                    #如果辨識是符號則捨棄
                    if word.split('_')[0]=='inter':
                        continue

                    #小寫的類別名為重疊，如aa/bb，取單一字母
                    if word.isalpha() and word.islower():
                        word=word[0]

                    #l_ee -> l_e
                    elif ('_' in word) and word.split('_')[-1].islower():
                        word='l_'+word.split('_')[-1][0]

                    assign_null.loc[p,'predictText']=word.split('_')[-1]
                    
                    #以便concat
                    temp_assign_null=assign_null.loc[[p]][['filename','xmin','ymin','xmax','ymin','xmax','ymax','xmin','ymax','predictText']]
                    temp_assign_null.columns=['filename','p1_x','p1_y','p2_x','p2_y','p3_x','p3_y','p4_x','p4_y','predictText']

                    results_str_df=pd.concat([results_str_df,temp_assign_null])
        
        #將此filename的圖片結果由results_str_df儲存至最終df中 注：index会保留为每个results_str_df時的index
        df=pd.concat([df,results_str_df])

        #print(results_str_df)
        #print('next')

    #將坐標轉換為整數
    for i in range(df.shape[0]):
        for j in range(1,9):
            df.iloc[i,j]=str(round(df.iloc[i,j]))

    df.to_csv(os.path.join(output_path,'submission.csv'),index=False,header=False,encoding="utf_8")
    #break
