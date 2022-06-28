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
import os

batchsz=50
validation_ds=ValidationSpectrogram(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test0617\spe4bands12.txt', transform=data_transforms)
validation_ds.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test0617\spe4bands12.txt', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(dataset=validation_ds,
                                              batch_size=batchsz,
                                              shuffle=False)
#"E:\SeaIceClassification\models\modelp1test0616.pkl"
#"E:\SeaIceClassification\models\models12test0625fff.pkl"
device = torch.device('cuda')
idtxt = open('img_txt\models12test0625.txt', 'w')

model = torch.load('models\models12test0625fff.pkl')
model.eval()
with torch.no_grad(),open('img_txt\models12test0625.txt','a') as f:
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        pprreedd = np.array(pred.cpu())
        np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')


idtxt.close()