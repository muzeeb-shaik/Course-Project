from PIL import Image, ImageOps
import PIL
import cv2
import numpy as np
import pandas as pd
import os

sz = 64*64
fname = []
for i in range(sz):
    fstr = "feature" + str(i+1)
    fname.append(fstr)
fname.append("label")
df = pd.DataFrame(columns=fname)
#df.to_csv('fname.txt')

impath = os.getcwd()
impath1 = os.path.join(impath, "COVID-19 Radiography Database\\COVID-19")
impath2 = os.path.join(impath, "COVID-19 Radiography Database\\NORMAL")
impath3 = os.path.join(impath, "COVID-19 Radiography Database\\Viral Pneumonia")
#impath1 = "C:\\GIT\\HW4\\covid19-radiography-database\\COVID-19 Radiography Database\\COVID-19"
#impath2 = "C:\\GIT\\HW4\\covid19-radiography-database\\COVID-19 Radiography Database\\NORMAL"
currpath = os.getcwd()
dir1 = "Covid64"
svpath1 = os.path.join(currpath, dir1)
if(os.path.isdir(svpath1) == False):
    os.mkdir(svpath1)
dir2 = "Normal64"
svpath2 = os.path.join(currpath, dir2)
if(os.path.isdir(svpath2) == False):
    os.mkdir(svpath2)
dir3 = "Pneumonia64"
svpath3 = os.path.join(currpath, dir3)
if(os.path.isdir(svpath3) == False):
    os.mkdir(svpath3)

filedata = os.listdir(impath1)
n = 0
j = 0
for f in filedata:
    pix_val = []
    i_path = os.path.join(impath1, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    for p in range(rows):
        for q in range(cols):
             pix_val.append(im1[p, q])
    print("Processing Image ",j)
    sv = svpath1 + "\\image" + str(j+1) + ".jpg"
    svimg = cv2.imwrite(sv,im1)
    pix_val.append(1)
    df.loc[j] = pix_val
    n += 1
    j += 1
    #if(n == 5):
    #   break

print("covid 19 feature extracted")
print(df)

filedata = os.listdir(impath2)
n = 0
for f in filedata:
    pix_val = []
    i_path = os.path.join(impath2, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    for p in range(rows):
        for q in range(cols):
             pix_val.append(im1[p, q])
    print("Processing Image ",j)
    sv = svpath2 + "\\image" + str(j+1) + ".jpg"
    svimg = cv2.imwrite(sv,im1)
    pix_val.append(-1)
    df.loc[j] = pix_val
    n += 1
    j += 1
    #if(n == 5):
    #   break

filedata = os.listdir(impath3)
n = 0
for f in filedata:
    pix_val = []
    i_path = os.path.join(impath3, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    for p in range(rows):
        for q in range(cols):
             pix_val.append(im1[p, q])
    print("Processing Image ",j)
    sv = svpath3 + "\\image" + str(j+1) + ".jpg"
    svimg = cv2.imwrite(sv,im1)
    pix_val.append(1)
    #df.loc[j] = pix_val
    n += 1
    j += 1
    #if(n == 5):
    #   break

print(df)
df.to_csv('image_features.txt',index=False)

