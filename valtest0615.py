import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from pauliDataset import pauliDataset
from ValidationSpectrogram import ValidationSpectrogram
from ValidationPauli import ValidationPauli
from net_test import CNeuralNetwork
from osgeo import gdal
from pauliDataset import data_transforms
import numpy as np
from torch.nn import functional as F
import os

batchsz=50
validation_ds=ValidationPauli(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP201761520\pauli48.txt', transform=data_transforms)
validation_ds.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP201761520\pauli48.txt', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(dataset=validation_ds,
                                              batch_size=batchsz,
                                              shuffle=False)

device = torch.device('cuda')
#"E:\SeaIceClassification\models\modelp36test0621_all.pkl"
#"E:\SeaIceClassification\models\modelp48test0626_all.pkl"

idtxt = open('img_txt\ALPSRP201761520p1_48.txt', 'w')
#porb=open('probability\\ALPSRP258351560test0625.txt','w')
#losstxt=open('probability\\losstest0625.txt','w')
#open('probability\\test0625.txt','a') as ff,open('probability\\losstest0625.txt','a') as fff



model = torch.load('models\modelp48test0626_all.pkl')


loss_func=nn.CrossEntropyLoss().to(device)
model.eval()
with torch.no_grad(),open('img_txt\ALPSRP201761520p1_48.txt','a') as f:
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
       # lloogg = np.array(logits.cpu())
        pred = logits.argmax(dim=1)
       # loss = F.softmax(logits,dim=1)
        #lossloss=np.array(loss.cpu(),dtype=float)
        pprreedd = np.array(pred.cpu())
      #  np.savetxt(fff,lossloss, fmt='%d',delimiter=' ')
        np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')
       # np.savetxt(ff, lloogg, fmt='%d', delimiter=' ')

idtxt.close()