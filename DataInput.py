#NOTE: This takes a while, this program takes both datasets in the folders 0 and 1 and linerizes the data (in greyscale) 
#NOTE: To Run this it took me 31 minutes

import numpy as np
import os
from PIL import Image
import random
import math
import time

start_time = time.time()
SIZE = 101*101 #Size of the images 

#Load all images in Data set 0 into a list
print("LOADING WORM DATA SET 0")
imgs = []
path = os.getcwd() + "/C. elegans/Data/0"
valid_images = [".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array NoWorm
print("FORMATTING LIST INTO NUMPY ARRAY")
NoWorm = np.empty((0,SIZE),int)
for i in range(len(imgs)):
    temp = np.reshape(imgs[i].convert('L'),SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    NoWorm  = np.append(NoWorm,temp,0)
print("Shape of Formatted Numpy Array: ", NoWorm.shape)

#Load all images in Data set 1 into a list
print("LOADING WORM DATA SET 1")
imgs = []
path = os.getcwd() + "/C. elegans/Data/1"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))

#Take all images loaded into list and convert them into greyscale, reshape them to be 1 x SIZE, then append them to numpy array Worm
print("FORMATTING LIST INTO NUMPY ARRAY")
Worm = np.empty((0,SIZE),int)
for i in range(len(imgs)):
    temp = np.reshape(imgs[i].convert('L'),SIZE)
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