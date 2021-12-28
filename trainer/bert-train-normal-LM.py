from transformers import AutoModelForMaskedLM, AutoTokenizer,BertTokenizer
import torch
from datasets import load_dataset,Dataset
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch import nn
import random
import numpy as np
import gc

#將字串遮住(以[MASK]代替),訓練一個模型使其可以判斷[MASK]是哪個字的幾率最高

dataset_path=r'C:\Users\yaoching\Desktop\transformer\tfer-dataset.csv'
model_save=r'C:\Users\yaoching\Desktop\transformer\model_saved'
num_epochs = 150
training_batch_size=20
eval_batch_size=4

#為eval的bacth加mask 
def mask_for_eval(bacth):
    
    #bathc中的每句話都隨機一個mask位置
    bacth['length']=bacth['length'].view(-1)

    #回傳每個batch的label 0:數字 1:英文 2:中文
    labels=[]
    #對batch中的每個樣本
    for i in range(bacth['length'].shape[0]):

        labels.append([-100]*bacth['input_ids'].shape[1])

        #隨機選擇一個字
        #pick=random.randint(1,bacth['length'][i])
        #暫時固定選第二個位置
        pick=2

        #將挑選的字之id給labels
        labels[i][pick]=bacth['input_ids'][i][pick].item()
        
        #替換成mask
        bacth['input_ids'][i,pick]=tokenizer.encode(tokenizer.mask_token)[1]
    
    
    labels=torch.tensor(labels,dtype=torch.long)
    
    #print(tokenizer.batch_decode(bacth['input_ids'].cpu().numpy().tolist()))

    #batch['input_id']已經隨機mask完畢，labels:[[-100,-100,-100,class_label,-100],...]
    return bacth,labels


#random mask 
def random_mask(bacth):
    
    #bathc中的每句話都隨機一個mask位置
    bacth['length']=bacth['length'].view(-1)

    #回傳每個batch的label 0:數字 1:英文 2:中文
    labels=[]
    for i in range(bacth['length'].shape[0]):

        labels.append([-100]*bacth['input_ids'].shape[1])

        #隨機選擇一個字
        pick=random.randint(1,bacth['length'][i])
        
        #將挑選的字之id給labels
        labels[i][pick]=bacth['input_ids'][i][pick].item()
        
        #替換成mask
        bacth['input_ids'][i,pick]=tokenizer.encode(tokenizer.mask_token)[1]
    
    
    labels=torch.tensor(labels,dtype=torch.long)
    
    #print(tokenizer.batch_decode(bacth['input_ids'].cpu().numpy().tolist()))

    #batch['input_id']已經隨機mask完畢，labels:[[-100,-100,-100,class_label,-100],...]
    return bacth,labels

def tokenize_function(examples):
    #example["text"]:['xxx','xxx','xxx','xxx',...] --> [['x','x','x'],['x','x''x'],...]
    #這樣做才可以實現characterlevel tokeizer

    #保存每個字串的長度，給後續的random_mask使用
    length=[]

    #這樣做才可以實現characterlevel tokeizer
    for i in range(len(examples["text"])):
        examples["text"][i]=list(examples["text"][i])
        length.append([len(examples["text"][i])])

    output=tokenizer(examples["text"], padding="max_length", max_length=512,truncation=True,is_split_into_words=True)
    
    output['length']=length

    return output

#加載用於Masked language modeling的模型和tokenizer,注意do_lower_case需給定False
tokenizer = BertTokenizer(vocab_file=r"C:\Users\yaoching\Desktop\transformer\tokens.txt",do_lower_case=False)
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

#根据新的vocab改變模型尺寸
model.resize_token_embeddings(len(tokenizer))

#修改輸出類別數
#model.vocab_projector=nn.Linear(768,3,bias=True)

#resume之前訓練到一半的
#model.load_state_dict(torch.load(r'C:\Users\yaoching\Desktop\transformer\model_saved\pytorch_model.bin'))

#加載資料集
#直接用load_dataset沒辦法處理None值問題
#dataset = load_dataset('csv', data_files=r'C:\Users\yaoching\Desktop\transformer\tfer-dataset.csv')
dataset = pd.read_csv(dataset_path)

#防止資料集裡包含'NA'字串會讀成None,後續tokenizer會報錯
dataset = dataset.dropna()

#拆分訓練集&驗證集
dataset_train = Dataset.from_pandas(dataset[:70000])
dataset_eval = Dataset.from_pandas(dataset[70000:])

#tokenizer化
tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
tokenized_datasets_eval = dataset_eval.map(tokenize_function, batched=True)

tokenized_datasets_train = tokenized_datasets_train.remove_columns(["__index_level_0__"])
tokenized_datasets_train = tokenized_datasets_train.remove_columns(["text"])
tokenized_datasets_train = tokenized_datasets_train.remove_columns(["token_type_ids"])
#將值轉換為tensor
tokenized_datasets_train.set_format("torch")

tokenized_datasets_eval = tokenized_datasets_eval.remove_columns(["__index_level_0__"])
tokenized_datasets_eval = tokenized_datasets_eval.remove_columns(["text"])
tokenized_datasets_eval = tokenized_datasets_eval.remove_columns(["token_type_ids"])
#將值轉換為tensor
tokenized_datasets_eval.set_format("torch")


train_dataloader = DataLoader(tokenized_datasets_train, shuffle=True, batch_size=training_batch_size)
eval_dataloader = DataLoader(tokenized_datasets_eval, batch_size=eval_batch_size)

optimizer = AdamW(model.parameters(), lr=5e-5)


num_training_steps = len(train_dataloader)
num_eval_steps =len(eval_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=3,
    num_training_steps=num_epochs * len(train_dataloader)
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#用來判斷是否要保存模型
max_accuracy=0

for epoch in range(num_epochs):

    #進度條
    progress_bar = tqdm(range(num_training_steps),desc=f'training epoch {epoch}')

    #計算每個epoch的loss
    temp_loss=0
    iter_num=0
    model.train()
    for batch in train_dataloader:
        
        #將batch中每句話隨機遮住一個字符，以期判斷它是數字、英文還是中文
        batch,labels=random_mask(batch)

        #刪除長度
        del batch['length']
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch,labels=labels.cuda())
      
        loss = outputs.loss
        loss.backward()
        
        #記錄loss
        temp_loss+=loss.item()
        
        iter_num+=1
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
        #防止內存爆掉
        del batch,outputs,labels
        gc.collect()
        
        '''
        #先跑幾個iter試試水
        if iter_num>100:
            break
        '''

    del progress_bar
    gc.collect()

    print(f'\n epoch:{epoch} loss:{round(temp_loss/iter_num,4)} \n')


    #進度條
    progress_eval_bar = tqdm(range(num_eval_steps),desc=f'eval epoch {epoch}')

    #計算準確率
    model.eval()
    sample_num=0
    correct=0
    accuracy=0.
    for batch in eval_dataloader:
        
        #將batch中每句話隨機遮住一個字符，以期判斷它是數字、英文還是中文
        batch,labels=mask_for_eval(batch)
        
        #刪除長度
        del batch['length']
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        result=outputs.logits

        #只看第二個預測結果(因為我們只給了它MASK)
        result=result[:,2]
        result=result.argmax(dim=1)

        #對的總樣本數
        correct+=np.sum((result==labels[:,2].cuda()).cpu().numpy())
    
        #總共eval的樣本數
        sample_num+=batch['input_ids'].shape[0]
        progress_eval_bar.update(1)
        #防止內存爆掉
        del batch,outputs,labels,
        gc.collect()
    
    del progress_eval_bar
    gc.collect()

    accuracy=correct/sample_num
    print(f'\n epoch:{epoch} accurary:{round(accuracy,4)} \n')

    if accuracy>max_accuracy:
        #保存模型
        max_accuracy=accuracy
        model.save_pretrained(model_save)
        print(f'model saved at {model_save} with accuracy {max_accuracy}!')
print(f'\n Finally accuracy:{max_accuracy}')