from PIL import Image, ImageOps
import PIL
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
dir1 = "Covid64_changed_orient"
svpath1 = os.path.join(currpath, dir1)
if(os.path.isdir(svpath1) == False):
    os.mkdir(svpath1)
dir2 = "Normal64_changed_orient"
svpath2 = os.path.join(currpath, dir2)
if(os.path.isdir(svpath2) == False):
    os.mkdir(svpath2)
dir3 = "Pneumonia64_changed_orient"
svpath3 = os.path.join(currpath, dir3)
if(os.path.isdir(svpath3) == False):
    os.mkdir(svpath3)

filedata = os.listdir(impath1)
n = 0
j = 0
for f in filedata:

    i_path = os.path.join(impath1, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    rot = 0
    while(rot<8):
        if rot <=3:
            pix_val_1 = []
            for p in range(rows):
                for q in range(cols):
                     pix_val_1.append(im1[p, q])
            print("Processing Image ",j)
            sv = svpath1 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im1)
            pix_val_1.append(1)
            df.loc[j] = pix_val_1
            n += 1
            j += 1

            im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)

        else:
            pix_val_2 = []
            im2 = cv2.flip(im1, 1)
            for p in range(rows):
                for q in range(cols):
                     pix_val_2.append(im2[p, q])
            print("Processing Image ",j)
            sv = svpath1 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im2)
            pix_val_2.append(1)
            df.loc[j] = pix_val_2
            n += 1
            j += 1
            im2 = cv2.flip(im2,1)
            im1 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)

        rot+=1
    #if(n > 20):
    #    break

print("covid 19 feature extracted")
print(df)

filedata = os.listdir(impath2)
n = 0
for f in filedata:

    i_path = os.path.join(impath2, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    rot = 0
    while(rot<8):
        if rot <=3:
            pix_val_1 = []
            for p in range(rows):
                for q in range(cols):
                     pix_val_1.append(im1[p, q])
            print("Processing Image ",j)
            sv = svpath2 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im1)
            pix_val_1.append(0)
            df.loc[j] = pix_val_1
            n += 1
            j += 1

            im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)

        else:
            pix_val_2 = []
            im2 = cv2.flip(im1, 1)
            for p in range(rows):
                for q in range(cols):
                     pix_val_2.append(im2[p, q])
            print("Processing Image ",j)
            sv = svpath2 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im2)
            pix_val_2.append(0)
            df.loc[j] = pix_val_2
            n += 1
            j += 1
            im2 = cv2.flip(im2,1)
            im1 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)

        rot+=1
    #if(n > 20):
    #    break

print("Normal feature extracted")
print(df)

filedata = os.listdir(impath3)
n = 0
for f in filedata:

    i_path = os.path.join(impath3, f)
    img = cv2.imread(i_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(gray_img, (64, 64))
    rows, cols = im1.shape
    rot = 0
    while(rot<8):
        if rot <=3:
            pix_val_1 = []
            for p in range(rows):
                for q in range(cols):
                     pix_val_1.append(im1[p, q])
            print("Processing Image ",j)
            sv = svpath3 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im1)
            pix_val_1.append(2)
            df.loc[j] = pix_val_1
            n += 1
            j += 1

            im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)

        else:
            pix_val_2 = []
            im2 = cv2.flip(im1, 1)
            for p in range(rows):
                for q in range(cols):
                     pix_val_2.append(im2[p, q])
            print("Processing Image ",j)
            sv = svpath3 + "\\image" + str(j+1) + ".jpg"
            svimg = cv2.imwrite(sv,im2)
            pix_val_2.append(2)
            df.loc[j] = pix_val_2
            n += 1
            j += 1
            im2 = cv2.flip(im2,1)
            im1 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)

        rot+=1
    #if(n > 20):
    #    break

print("Pneumonia64 feature extracted")
print(df)

df.to_csv('image_features_orientation.txt',index=False)
