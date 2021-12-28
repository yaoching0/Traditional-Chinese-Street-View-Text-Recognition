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
