import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from pauliDataset import pauliDataset
from ValidationSpectrogram import ValidationSpectrogram
from ValidationPauli import ValidationPauli
from net_test36 import CNeuralNetwork
from osgeo import gdal
from pauliDataset import data_transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


#"E:\ALOSPALSAR\TrainData\ALPSRP205991510\ALPSRP205991510_36.txt"
#"E:\ALOSPALSAR\TrainData\ALPSRP205991510\ALPSRP205991510_48.txt"
# writer = SummaryWriter('logs')
#"E:\SeaIceClassification\pauliDataPath36.txt"
batchsz=50
pauli_ds=pauliDataset(labeltxt='pauliDataPath36.txt',transform=data_transforms)
pauli_ds.__init__(labeltxt='pauliDataPath36.txt', transform=data_transforms)



train_loader = torch.utils.data.DataLoader(dataset=pauli_ds,
                                               batch_size=batchsz,
                                               shuffle=True)
device = torch.device('cuda')

x, label = iter(train_loader).next()
print('x:', x.shape, 'label:', label.shape)

model = CNeuralNetwork().to(device)

loss_func=nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(model)

for epoch in range(1000):
    model.train()
    for batchidx, (x, label) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)
        #print(x.shape)
        logits = model(x)
        loss = loss_func(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # writer.add_scalar("Train/Loss", loss.item(), batchidx)
torch.save(model, 'models/modelp36test0621_all.pkl')


#model = torch.load('model.pkl')