#NOTE: This takes a while, this program takes both datasets in the folders 0 and 1 and linerizes the data (in greyscale) 
#NOTE: To Run this it took me 31 minutes

import numpy as np
import os
from PIL import Image
import random
import math
import time
import glob
import cv2 as cv

start_time = time.time()

path = "C:/Users/ajdan/Documents/PR-ML/Pattern Recognition/Project 4 PR/Pattern-Recognition-Project4/C. elegans/Data/0"

SIZE = 28*28
filenames = glob.glob(path + '/*.png')
filenames.sort()
images = [cv.imread(img) for img in filenames]

scale_percent = 28 # percent of original size
width = int(101 * scale_percent / 100)
height = int(101 * scale_percent / 100)
dim = (width, height)

imagesResized = [cv.resize(img, dim, interpolation=cv.INTER_AREA) for img in images]
imagesGrayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imagesResized]
imagesBlurred = [cv.GaussianBlur(img, (3,3), 0) for img in imagesGrayscale]

sobelx = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) for img in imagesBlurred]
sobely = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) for img in imagesBlurred]
sobelxy = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) for img in imagesBlurred]

edges = [cv.Canny(image=img, threshold1=60, threshold2=140) for img in imagesBlurred]
#Load all images in Data set 0 into a list
print("LOADING WORM DATA SET 0")


#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array NoWorm
print("FORMATTING LIST INTO NUMPY ARRAY")
NoWorm = np.empty((0,SIZE),int)
for i in range(len(edges)):
    temp = np.reshape(edges[i],SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    NoWorm  = np.append(NoWorm,temp,0)
print("Shape of Formatted Numpy Array: ", NoWorm.shape)


path = "C:/Users/ajdan/Documents/PR-ML/Pattern Recognition/Project 4 PR/Pattern-Recognition-Project4/C. elegans/Data/1"


SIZE = 28*28
filenames = glob.glob(path + '/*.png')
filenames.sort()
images = [cv.imread(img) for img in filenames]

scale_percent = 28 # percent of original size
width = int(101 * scale_percent / 100)
height = int(101 * scale_percent / 100)
dim = (width, height)

imagesResized = [cv.resize(img, dim, interpolation=cv.INTER_AREA) for img in images]
imagesGrayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imagesResized]
imagesBlurred = [cv.GaussianBlur(img, (3,3), 0) for img in imagesGrayscale]

sobelx = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) for img in imagesBlurred]
sobely = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) for img in imagesBlurred]
sobelxy = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) for img in imagesBlurred]

edges = [cv.Canny(image=img, threshold1=60, threshold2=140) for img in imagesBlurred]

#Load all images in Data set 1 into a list
print("LOADING WORM DATA SET 1")

#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array Worm
print("FORMATTING LIST INTO NUMPY ARRAY")
Worm = np.empty((0,SIZE),int)
for i in range(len(edges)):
    temp = np.reshape(edges[i],SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    Worm  = np.append(Worm,temp,0)
print("Shape of Formatted Numpy Array: ",Worm.shape)

#Stack both the Worms Data and No Worms Data
print("Formatting Data")
NumPic , Dim = NoWorm.shape
X = np.vstack((NoWorm,Worm))
Y = np.hstack((np.ones(NumPic),np.ones(NumPic)*0))
D = SIZE
K = 2

#Randomize the Order of the Data, generate DataX and DataT
DataX = np.empty((0,D),float)
DataT = np.empty((0,K),int)

Remain = (NumPic * 2) - 1
for i in range((NumPic*2)):
    p = random.randint(0,Remain)

    DataX = np.append(DataX,np.array([X[p]]),0)

    if Y[p] == 1:
        DataT = np.append(DataT,np.array([[1,0]]),0)
    elif Y[p] == 0:
        DataT = np.append(DataT,np.array([[0,1]]),0)

    X = np.delete(X,p,0)
    Y = np.delete(Y,p,0)
    Remain = Remain - 1
    if i % 1000 == 0:
        print("Current Sample: " , i)

#Normalize the Data, append column of 1
DataX = DataX / 255
DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

print("Data Sucessfully Randomized and Formated")
print(DataX.shape)
print(DataT.shape)

#Save the Data in npz format
print("Saving Data Matrices")
np.savez_compressed('DataX.npz', DataX = DataX)
np.savez_compressed('DataT.npz', DataT = DataT)

#Print Time it took 
print("Total Time Elapsed:", (time.time() - start_time))