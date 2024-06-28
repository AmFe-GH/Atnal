import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from importlib import reload
reload(nn)


class MSIdataset(Dataset):
    def __init__(self, data, xLoc, yLoc, zLoc):
        self.len = len(data)
        self.data = data
        self.xLoc = xLoc
        self.yLoc = yLoc
        self.zLoc = zLoc

    def __getitem__(self, index):
        return self.data[index], self.xLoc[index], self.yLoc[index], self.zLoc[index]

    def __len__(self):
        return self.len


class Model_trans(nn.Module):
    def __init__(self, d_mz, d_model=256, encoder_layer_num=7, use_decoder=False, decoder_layer_num=7, n_head=8, device='cpu'):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_mz, 512, device=device), nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3), nn.ReLU(),
                                      nn.Linear(512, d_model, device=device), nn.BatchNorm1d(num_features=d_model, momentum=0.99, eps=1e-3))
        # self.transformer=nn.Transformer(d_model,n_head,encoder_layer_num,decoder_layer_num,device=device)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head)
        self.transformerencoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=encoder_layer_num)
        # self.row=nn.parameter(torch.rand(batch_size//2,d_model//2))
        # self.col=nn.parameter(torch.rand(batch_size//2,d_model//2))
        # self.recover=nn.Sequential(nn.Linear(d_model,1024),nn.BatchNorm1d(num_features=1024),nn.ReLU(),nn.Linear(1024,d_mz),nn.Sigmoid())
        self.recover = nn.Sequential(nn.Linear(d_model, 512), nn.BatchNorm1d(num_features=512, momentum=0.99, eps=1e-3),
                                     nn.ReLU(), nn.Linear(512, d_mz), nn.Softmax(dim=-1))
        self.d_model = d_model
        self.position_Enbedding = None
        self.log = {'training loss': [],
                    'testing loss': []}

    def forward(self, input):
        h = self.backbone(input)
        # if self.position_Enbedding==None:
        # position_Enbedding=PositionalEncoding2d(d_model=self.d_model,xLocation= xLocation,yLocation=yLocation)
        # transformer_input=position_Enbedding+h
        transformer_input = h
        # #print(transformer_input.shape)
        # #print(self.query_pos.shape)
        h = self.transformerencoder(transformer_input)
        return self.TIC_norm(self.recover(h))

    def TIC_norm(self, input):
        return input/torch.sum(input, dim=-1)[:, None]


def categorical_crossentropy(pred, label):
    """
    使用pytorch 来实现 categorical_crossentropy
    """
    # print(-label * torch.log(pred))
    # print(-label * torch.log(pred))
    # pred=pred/(pred.sum(-1).reshape(-1,1))
    pred = torch.clip(pred, min=1e-7, max=1.-1e-7)
    loss = (torch.sum(-label * torch.log(pred)))/pred.shape[0]

    # return nn.Sigmoid()(loss)
    # loss = nn.BCELoss()(pred, label)
    # loss=nn.CrossEntropyLoss(pred,label)
    return loss
