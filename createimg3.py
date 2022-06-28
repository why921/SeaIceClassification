import numpy as np
import cv2


data1 = np.loadtxt('img_txt\stest0624.txt')
data2 = np.loadtxt('img_txt\modelp36test0621_all.txt')






imgdata1 = data1.reshape((300, 300))
imgdata2 = data2.reshape((300, 300))

output = np.zeros((300, 300, 3))  # 新建三维数组，且初始值为1
print(output[1][1])
output[1][1] = [255, 255, 255]
print(output[1][1])

class1 = np.where(imgdata1 == 0)
class2 = np.where(imgdata1 == 1)
class3 = np.where(imgdata2 == 2)
class4 = np.where(imgdata2 == 3)
class5 = np.where(imgdata2 == 4)

class44=np.where(imgdata2 == 3)
class33=np.where(imgdata1 == 2)
print(class1)
print(len(class1[0]))
print(len(class2[1]))
print(len(class3[1]))
print(class1[0])
for i in range(len(class1[0])):
    output[class1[0][i]][class1[1][i]] = [0, 0, 255]

for i in range(len(class2[0])):
    output[class2[0][i]][class2[1][i]] = [0, 255, 0]

for i in range(len(class3[0])):
    output[class3[0][i]][class3[1][i]] = [255, 0, 0]
for i in range(len(class33[0])):
    output[class33[0][i]][class33[1][i]] = [255, 0, 0]

for i in range(len(class4[0])):
    output[class4[0][i]][class4[1][i]] = [0, 255, 0]


for i in range(len(class5[0])):
    output[class5[0][i]][class5[1][i]] = [0, 255, 255]

cv2.imshow('123',output)
cv2.waitKey()