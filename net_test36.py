import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



class CNeuralNetwork(nn.Module):
    def __init__(self):
        super(CNeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        #2 32 6 6
        self.fc = nn.Sequential(
            nn.Linear(512*5*5,4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 5),
        )
    def forward(self, x):
        #b 3 24 24---b 32 6 6
        x = self.conv1(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output



def main():
    net=CNeuralNetwork()
    tmp = torch.randn(100, 3, 36, 36)
    out = net(tmp)
    print(out.shape)



if __name__ == '__main__':
    main()


#self.fc = nn.Linear(32 * (img_size // 4) * (img_size // 4), num_classes)
#"E:\SeaIceClassification\runs\Jun10_10-32-19_WHY-Y7000P\events.out.tfevents.1654828341.WHY-Y7000P.27096.0"