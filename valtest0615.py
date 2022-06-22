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
validation_ds=ValidationPauli(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli36.txt', transform=data_transforms)
validation_ds.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli36.txt', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(dataset=validation_ds,
                                              batch_size=batchsz,
                                              shuffle=False)
#"E:\SeaIceClassification\models\modelp1test0616.pkl"
device = torch.device('cuda')
idtxt = open('img_txt\modelp36test0621_all.txt', 'w')
model = torch.load('models\modelp36test0621_all.pkl')
model.eval()
with torch.no_grad(),open('img_txt\modelp36test0621_all.txt','a') as f:
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        pprreedd = np.array(pred.cpu())
        np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')
idtxt.close()