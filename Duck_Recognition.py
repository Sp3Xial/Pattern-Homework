import os
import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB



duck_sample = []
duck_lable = []



isduck_image = os.listdir('duck/')
for isduck in isduck_image:
    imagename = "duck/" + isduck
    img = cv2.imread(imagename,1)
    info = img.shape
    imgH = info[0]
    imgW = info[1]
    for i in range(imgH):
        for j in range(imgW):
            duck_sample.append(img[i][j])
            duck_lable.append(0)



non_duck_image = os.listdir('non_duck/')
for nonduck in non_duck_image:
    imagename = "non_duck/" + nonduck
    img = cv2.imread(imagename,1)
    info = img.shape
    imgH = info[0]
    imgW = info[1]
    for i in range(imgH):
        for j in range(imgW):
            duck_sample.append(img[i][j])
            duck_lable.append(1)



train_data = np.array(duck_sample)
lable_data = np.array(duck_lable)



clf_pf = GaussianNB()
clf_pf.fit(train_data,lable_data)



image = cv2.imread("full_duck.jpg",1)
imageinfo = image.shape
imgH = imageinfo[0]
imgW = imageinfo[1]
imgD = imageinfo[2]
newinfo = (imgH,imgW,imgD)
dst = np.zeros(newinfo,np.uint8)


for i in range(0,imgH):
    for j in range(0,imgW):
        dst[i,j] = image[i,j]

for i in range(0,imgH):
    for j in range(0,imgW):
        (b,g,r) = image[i,j]
        blue = int(b)
        green = int(g)
        red = int(r)
        if clf_pf.predict([dst[i][j]]) == 0:
            dst[i,j] = (0,0,255)

cv2.imwrite("marked_full_duck.jpg",dst)
print("completed")