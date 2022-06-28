import numpy as np
import cv2


#"E:\SeaIceClassification\probability\losstest0625.txt"
data = np.loadtxt('probability\\losstest0625.txt')

print(data.shape)

imgdata = data.reshape((300, 300, 5))
oo=imgdata[:,:,0]
print(oo.shape)
probdata=np.zeros((300, 300))

for i in range(0,5):
  probdata+=imgdata[:,:,i]


cv2.imshow('123',probdata)
cv2.waitKey()

print(imgdata)