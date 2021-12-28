import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm

#訓練用於模型融合的EnsembleNet

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

ensemble_feature_test_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'ensemble_feature.csv')

#切分數據，驗證集為整個數據集的四分之一
traning_data=pd.read_csv(ensemble_feature_test_path,header=None)
val_data=traning_data.iloc[range(0,traning_data.shape[0],4),:]
traning_data=traning_data.iloc[list(set(range(0,traning_data.shape[0])).difference(set(range(0,traning_data.shape[0],4)))),:]
print('traning data number: %d , val data number: %d '%(traning_data.shape[0],val_data.shape[0]))

#输入data的feature數量
num_inputs = traning_data.shape[1]-1
#print(num_inputs)

features_train=torch.from_numpy(traning_data.iloc[:,:-1].values).type(torch.float32).cuda() 
labels_train=torch.from_numpy(traning_data.iloc[:,-1].values).type(torch.float32).cuda() 

features_val=torch.from_numpy(val_data.iloc[:,:-1].values).type(torch.float32).cuda() 
labels_val=torch.from_numpy(val_data.iloc[:,-1].values).type(torch.float32).cuda() 

#print(features.shape)
#print(labels.shape)

batch_size = 16

dataset_train = Data.TensorDataset(features_train, labels_train)
data_iter_train = Data.DataLoader(dataset_train, batch_size, shuffle=True)

dataset_val = Data.TensorDataset(features_val, labels_val)
data_iter_val = Data.DataLoader(dataset_val, batch_size, shuffle=True)

#嘗試各種網路架構
'''
net = nn.Sequential(
    FlattenLayer(),
    #nn.Linear(num_inputs, num_hidden),
    nn.Linear(num_inputs, 5122),
    #nn.ReLU(),
    #nn.Linear(num_hidden, 5122),
    nn.Sigmoid(),
    #如果loss用nn.CrossEntropyLoss()則不需要，但最後inference階段還需要再加一個softmax
    #nn.Softmax(dim=1)
    )
'''

'''
class EnsembleNet(nn.Module):
    def __init__(self, **kwargs):
        super(EnsembleNet, self).__init__(**kwargs)
        self.linear=nn.Linear(num_inputs, 5122)
        self.act=nn.Sigmoid()

    def forward(self, x):
        output= self.linear(x)
        output=output+x[:,:5122]+x[:,5122:10244]+x[:,10244:]
        return self.act(output)
'''

class EnsembleNet(nn.Module):
    def __init__(self, **kwargs):
        super(EnsembleNet, self).__init__(**kwargs)
        self.params=nn.Parameter(torch.randn(5122,3))
        self.bias =nn.Parameter(torch.randn(5122))   
        self.ratio=nn.Parameter(torch.randn(3)) 
        self.act=nn.Sigmoid()
        
    def forward(self,x):
        
        v1=x[:,:5122]
        v2=x[:,5122:10244]
        v3=x[:,10244:]

        #shape:[batch,5122,3]
        V=torch.stack((v1,v2,v3),dim=2)

        #broadcast [shape:[batch,5122,3]]
        output=self.params*V 

        #shape:[batch,5122]
        output=torch.sum(output,dim=2)+self.bias

        #with殘差
        output=nn.functional.relu(output)+self.ratio[0]*x[:,:5122]+self.ratio[1]*x[:,5122:10244]+self.ratio[2]*x[:,10244:]
        
        return self.act(output)



'''
class EnsembleNet(nn.Module):
    def __init__(self, **kwargs):
        super(EnsembleNet, self).__init__(**kwargs)
        self.params=nn.Parameter(torch.randn(5122,3))
        self.bias =nn.Parameter(torch.randn(5122))
        self.ratio=nn.Parameter(torch.randn(3))
        self.act=nn.Sigmoid()

    def forward(self,x):
        
        v1=x[:,:5122]
        v2=x[:,5122:10244]
        v3=x[:,10244:]

        #shape:[batch,5122,3]
        V=torch.stack((v1,v2,v3),dim=2)

        #broadcast [shape:[batch,5122,3]]
        output=self.params*V 

        ##shape:[batch,5122]
        output=torch.sum(output,dim=2)+self.bias

        
        #with殘差
        normalized_ratio=self.ratio/(torch.sum(self.ratio,dim=0))
        #print(normalized_ratio)
        
        output=nn.functional.relu(output)+normalized_ratio[0]*x[:,:5122]+normalized_ratio[1]*x[:,5122:10244]+normalized_ratio[2]*x[:,10244:]

        return output
'''  

net=EnsembleNet()
net.cuda()


for params in net.parameters():
    init.normal_(params, mean=0., std=0.01)
#weight_decay=1e-04
optimizer = optim.Adam(net.parameters(), lr=0.0022)
#損失函數
#loss = nn.MSELoss()
loss =torch.nn.CrossEntropyLoss()
#print(optimizer.param_groups)
max_accuary=0
num_epochs =140 #144 training集100%擬合
for epoch in range(1, num_epochs + 1):
    if (epoch%5==0):
        optimizer.param_groups[0]['lr']*=0.99
    train_l_sum,n=0.0,0.0
    for X, y in tqdm(data_iter_train):
        output = net(X)
        #右邊必須是torch.long(即int64)
        l = loss(output, y.type(torch.long))
        optimizer.zero_grad() # 梯度清零
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        n += y.shape[0]

    #計算準確率
    num_all=0.
    num_cor=0.
    for input, target in data_iter_val:
        result=net(input).argmax(dim=1)
        #print(target,result)
        
        for i in range(target.shape[0]):
            num_all+=1
            if(result[i]==target[i]):
                num_cor+=1

    #在驗證集有準確率更高的權重就儲存起來
    if(num_cor/num_all>max_accuary):
        max_accuary=num_cor/num_all
        model_weight_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'ensemble_weights','model-weight-epoch{}-{:.2f}%.pth'.format(epoch,max_accuary*100))
        torch.save(net.state_dict(),model_weight_path)
    print('epoch %d, loss: %f val accuary: %.6f' % (epoch, train_l_sum/n,num_cor/num_all))

    #查看權重
    print(net.params[:3])
    print(net.ratio)
#隨機抽幾個樣本查看訓練效果
'''
input_test=torch.from_numpy(traning_data.iloc[0,:-1].values).type(torch.float32).unsqueeze(0).cuda()
print('output:',net(input_test).argmax(dim=1))
print('label:',traning_data.iloc[0,-1])

'''
#assert False

print("DONE")

