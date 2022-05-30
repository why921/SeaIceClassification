import numpy as np
from osgeo import gdal
from torchvision import transforms
from torch.utils.data import Dataset
import DataTrans

class ValidationSpectrogram(Dataset):
    def __init__(self, labeltxt, transform, target_transform=None):
        fh = open(labeltxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存
            # words[0]图片，words[1]lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn图片path #fn和label分别获得imgs[index]即每行中word[0]和word[1]的信息
        ds = gdal.Open(fn)
        band1 = ds.GetRasterBand(1)
        band2 = ds.GetRasterBand(2)
        band3 = ds.GetRasterBand(3)
        band4 = ds.GetRasterBand(4)

        im_data1 = band1.ReadAsArray()
        im_data2 = band2.ReadAsArray()
        im_data3 = band3.ReadAsArray()
        im_data4 = band4.ReadAsArray()
        img = np.array([im_data1, im_data2, im_data3,im_data4])

        if self.transform is not None:
            img = self.transform(img)
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


data_transforms = transforms.Compose([
   DataTrans.Numpy2Tensor(),
])


ddss = ValidationSpectrogram(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt',transform=data_transforms)
ddss.__init__(labeltxt='E:\ALOSPALSAR\ValidationData\ALPSRP205991510test\spe4bands12.txt',transform=data_transforms)
print(ddss.__len__())
img, gt = ddss.__getitem__(2) # get the 34th sample
print(type(img))
print(img)
print(gt)