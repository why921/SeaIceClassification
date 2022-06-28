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
#"E:\ALOSPALSAR\TrainData\spe4bands12.txt"
    batchsz=50



    #"E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\pauli24.txt"
    validation_ds=ValidationSpectrogram(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP258351560_3\spe4bands12.txt', transform=data_transforms)
    validation_ds.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP258351560_3\spe4bands12.txt', transform=data_transforms)

#""E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt""

    test_loader = torch.utils.data.DataLoader(dataset=validation_ds,
                                              batch_size=batchsz,
                                              shuffle=False)
    device = torch.device('cuda')

    model = torch.load('models\models12test0625fff.pkl')
    idtxt = open('img_txt\ALPSRP258351560spe3.txt', 'w')
    model.eval()
    with torch.no_grad(),open('img_txt\ALPSRP258351560spe3.txt','a') as f:
        for x, label in test_loader:
          x, label = x.to(device), label.to(device)
          logits = model(x)
          pred = logits.argmax(dim=1)
          pprreedd = np.array(pred.cpu())
          np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')
    idtxt.close()

if __name__ == '__main__':
    main()

