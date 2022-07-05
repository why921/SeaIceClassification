import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from pauliDataset import pauliDataset
from ValidationSpectrogram import ValidationSpectrogram
from ValidationPauli import ValidationPauli
from snet_test import SNeuralNetwork
from spectrogramDataset import spectrogramDataset
from osgeo import gdal
from pauliDataset import data_transforms
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():

    batchsz=64
   #"E:\ALOSPALSAR\TrainData\spe4bands12.txt"
    spectrogram_ds = spectrogramDataset(labeltxt='E:\ALOSPALSAR\TrainData\spe4bands12.txt', transform=data_transforms)
    spectrogram_ds.__init__(labeltxt='E:\ALOSPALSAR\TrainData\spe4bands12.txt',transform=data_transforms)


    train_loader = torch.utils.data.DataLoader(dataset=spectrogram_ds,
                                               batch_size=batchsz,
                                               shuffle=True)

    device = torch.device('cuda')

    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    model = SNeuralNetwork().to(device)
    model.zero_grad()

    loss_func=nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)



    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = loss_func(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'models//s12model0701_1.pkl')

if __name__ == '__main__':
    main()
