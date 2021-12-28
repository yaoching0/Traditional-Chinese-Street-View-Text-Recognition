import json
import pprint
import os

#兩個路徑要執行兩次
train_anno=r'C:\Users\yaoching\Desktop\繁中場景\ctw-annotations.tar\products\ctw-annotations\train.jsonl'
val_anno=r'C:\Users\yaoching\Desktop\繁中場景\ctw-annotations.tar\products\ctw-annotations\val.jsonl'

output_path=r'C:\Users\yaoching\Desktop\繁中場景\ctw-annotations.tar\yolo_label'
#每個element都是一張照片的標註(dict格式)
anno_list=[]

with open(val_anno) as f:
    temp_list=f.readlines()
    for i in temp_list:
        anno_list.append(json.loads(i))
        

#對每張照片
for label in anno_list:

    #保存yolo label的路徑
    f_yolo=open(os.path.join(output_path,label['image_id']+'.txt'),"w")

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