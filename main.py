import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from spectrogramDataset import spectrogramDataset
from pauliDataset import pauliDataset
from net_test import CNeuralNetwork
from osgeo import gdal
from pauliDataset import data_transforms
import numpy as np
import os


def main():

    batchsz=100
    in_ds = SeaIceDataset(labeltxt='D:\why2022\seaice\data_process\green.txt', transform=data_transforms)
    in_ds.__init__(labeltxt='D:\why2022\seaice\data_process\green.txt', transform=data_transforms)

    ev_ds=evlDataset(labeltxt='D:\why2022\seaice\mi3w0409\ALOS-P1_1__A-ORBIT__ALPSRP256411570_Cal_ML_Spk_Decomppauli.txt', transform=data_transforms)
    ev_ds.__init__(labeltxt='D:\why2022\seaice\mi3w0409\ALOS-P1_1__A-ORBIT__ALPSRP256411570_Cal_ML_Spk_Decomppauli.txt', transform=data_transforms)


    train_loader = torch.utils.data.DataLoader(dataset=in_ds,
                                               batch_size=batchsz,
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=ev_ds,
                                              batch_size=batchsz,
                                              shuffle=False)
    device = torch.device('cuda')

    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    model = CNeuralNetwork().to(device)

   # loss_func = nn.MSELoss()
   #criteon = nn.CrossEntropyLoss().to(device)
    loss_func=nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    idtxt = open('test.txt', 'w')
    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            #print(x.shape)
            logits = model(x)
            loss = loss_func(logits, label - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
# print(epoch, 'loss:', loss.item())
    model.eval()
    with torch.no_grad(),open('test.txt','a') as f:
        for x, label in test_loader:
        # [b, 3, 32, 32]
        # [b]
          x, label = x.to(device), label.to(device)
          logits = model(x)
          pred = logits.argmax(dim=1)
          #pprreedd=pred.cpu().numpy()
          pprreedd = np.array(pred.cpu())
          #print(pprreedd)
          np.savetxt(f, pprreedd, fmt='%d',delimiter=' ')
          #idtxt.write(pprreedd)
    idtxt.close()
if __name__ == '__main__':
    main()

