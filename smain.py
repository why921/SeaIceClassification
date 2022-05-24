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

    batchsz=100
   # pauli_ds=pauliDataset(labeltxt='E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_24.txt',transform=data_transforms)
  #  pauli_ds.__init__(labeltxt='E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_24.txt', transform=data_transforms)

    spectrogram_ds=spectrogramDataset(labeltxt='E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_24_4bands.txt',transform=data_transforms)
    spectrogram_ds.__init__(labeltxt='E:\ALOSPALSAR\TrainData\ALPSRP267211510\ALPSRP267211510_spe_24_4bands.txt', transform=data_transforms)

#"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli24.txt"
    validation_ds=ValidationSpectrogram(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt', transform=data_transforms)
    validation_ds.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt', transform=data_transforms)

#""E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt""
    train_loader = torch.utils.data.DataLoader(dataset=spectrogram_ds,
                                               batch_size=batchsz,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=validation_ds,
                                              batch_size=batchsz,
                                              shuffle=False)
    device = torch.device('cuda')

    x, label = iter(train_loader).next()
 #   print('x:', x.shape, 'label:', label.shape)

    model = SNeuralNetwork().to(device)
    model.zero_grad()
   # loss_func = nn.MSELoss()
   #criteon = nn.CrossEntropyLoss().to(device)
    loss_func=nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
#    print(model)

    idtxt = open('stest.txt', 'w')
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
#print(epoch, 'loss:', loss.item())
    model.eval()
    with torch.no_grad(),open('stest.txt','a') as f:
        for x, label in test_loader:
          x, label = x.to(device), label.to(device)
          logits = model(x)
          pred = logits.argmax(dim=1)
          pprreedd = np.array(pred.cpu())
          np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')
    idtxt.close()

if __name__ == '__main__':
    main()

