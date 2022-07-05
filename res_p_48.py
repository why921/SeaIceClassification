import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from pauliDataset import pauliDataset
from ValidationSpectrogram import ValidationSpectrogram
from ValidationPauli import ValidationPauli
from res_p_48net import ResNet
from osgeo import gdal
from pauliDataset import data_transforms
import DataTrans
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#声明gpu
device=torch.device('cuda:0')#调用哪个gpu
#"E:\ALOSPALSAR\TrainData\ALPSRP205991510\ALPSRP205991510_36.txt"
#"E:\ALOSPALSAR\TrainData\ALPSRP205991510\ALPSRP205991510_48.txt"
# writer = SummaryWriter('logs')
#"E:\SeaIceClassification\pauliDataPath36.txt"
batchsz=32



def pTrans(degrees):
    Trans=transforms.Compose([DataTrans.Numpy2Tensor(),
        transforms.RandomRotation(degrees=[degrees,degrees]),
                               ])
    return Trans

pauli_ds0=pauliDataset(labeltxt='pauliDataPath48.txt',transform=pTrans(0))
pauli_ds0.__init__(labeltxt='pauliDataPath48.txt',transform=pTrans(0))
'''
pauli_ds90=pauliDataset(labeltxt='pauliDataPath48.txt',transform=pTrans(90))
pauli_ds90.__init__(labeltxt='pauliDataPath48.txt',transform=pTrans(90))

pauli_ds180=pauliDataset(labeltxt='pauliDataPath48.txt',transform=pTrans(180))
pauli_ds180.__init__(labeltxt='pauliDataPath48.txt',transform=pTrans(180))

pauli_ds270=pauliDataset(labeltxt='pauliDataPath48.txt',transform=pTrans(270))
pauli_ds270.__init__(labeltxt='pauliDataPath48.txt',transform=pTrans(270))

pauli_ds=torch.utils.data.ConcatDataset([pauli_ds0,pauli_ds90,pauli_ds180,pauli_ds270])
'''
train_loader = torch.utils.data.DataLoader(dataset=pauli_ds0,
                                               batch_size=batchsz,
                                               shuffle=True)

print(pauli_ds0.__len__())




x, label = iter(train_loader).next()
print('x:', x.shape, 'label:', label.shape)

model = ResNet().to(device)

loss_func=nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(model)

for epoch in range(100):
    print(epoch)
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
torch.save(model, 'models/resmodelp48test0701_all.pkl')