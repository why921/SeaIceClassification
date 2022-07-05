import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
#"E:\SeaIceClassification\img_txt\ALPSRP258351560spe2.txt"
data = np.loadtxt('img_txt\ALPSRP201761520spe1_12_0701.txt')

print(data.shape)

imgdata = data.reshape((300, 300))

print(imgdata.shape)

print(imgdata)
output = np.zeros((300, 300, 3))  # 新建三维数组，且初始值为1
print(output[1][1])
output[1][1] = [255, 255, 255]
print(output[1][1])

class1 = np.where(imgdata == 0)
class2 = np.where(imgdata == 1)
class3 = np.where(imgdata == 2)
class4 = np.where(imgdata == 3)
class5 = np.where(imgdata == 4)
print(class1)
print(len(class1[0]))
print(len(class2[1]))
print(len(class3[1]))
print(class1[0])
for i in range(len(class1[0])):
    output[class1[0][i]][class1[1][i]] = [0, 0, 0]
for i in range(len(class2[0])):
    output[class2[0][i]][class2[1][i]] = [0, 0, 255]
for i in range(len(class3[0])):
    output[class3[0][i]][class3[1][i]] = [255, 0, 0]
for i in range(len(class4[0])):
    output[class4[0][i]][class4[1][i]] = [0, 255, 0]
for i in range(len(class5[0])):
    output[class5[0][i]][class5[1][i]] = [0, 255, 255]
cv2.imshow('123',output)
cv2.waitKey()