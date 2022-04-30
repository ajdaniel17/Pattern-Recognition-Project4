import numpy as np
import glob
import cv2 as cv
import idx2numpy
import math

path = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/C. elegans/Data/0"

data = np.load('W_CElegans.npz')

W = data['W']
SIZE , K = W.shape

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

DataX = np.empty((0,SIZE),int)
for i in range(len(edges)):
    temp = np.reshape(edges[i],SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    DataX  = np.append(DataX,temp,0)

DataX = DataX / 255
DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

DataXTest = DataX[math.ceil(DataX.shape[0] * 0.8):]


yPred = np.dot(DataXTest, W)

totals = np.zeros((K))
for i in range(len(DataXTest)):
    totals[np.argmax(yPred[i])] += 1

print("Total Tallies For Worms 0:")
for i in range(K):
    print("Class", int(i) , ":" , int(totals[i]))


path = "C:/Users/ajdan/Documents/PR-ML/Machine Learning/Project 4 ML/Machine-Learning-Project4/C. elegans/Data/1"

data = np.load('W_CElegans.npz')

W = data['W']
SIZE , K = W.shape

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

DataX = np.empty((0,SIZE),int)
for i in range(len(edges)):
    temp = np.reshape(edges[i],SIZE)
    temp = temp.astype(int)
    temp = np.array([temp])
    DataX  = np.append(DataX,temp,0)

DataX = DataX / 255
DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

DataXTest = DataX[math.ceil(DataX.shape[0] * 0.8):]


yPred = np.dot(DataXTest, W)

totals = np.zeros((K))
for i in range(len(DataXTest)):
    totals[np.argmax(yPred[i])] += 1

print("Total Tallies For Worms 1:")
for i in range(K):
    print("Class", int(i) , ":" , int(totals[i]))



data = np.load('trainedModelMNIST.npz')

W = data['x']


imageTestFile = 'MNIST/DataUncompressed/test-images.idx3-ubyte'  # Load Test Images File
labelTestFile = 'MNIST/DataUncompressed/test-labels.idx1-ubyte'  # Load Test Labels File


imageTestArr = idx2numpy.convert_from_file(imageTestFile)  # Convert testing images to numpy arrays
labelTestArr = idx2numpy.convert_from_file(labelTestFile)  # Convert testing labels to numpy arrays

imageTestArrLinearized = imageTestArr.reshape(imageTestArr.shape[0], imageTestArr.shape[1] * imageTestArr.shape[2])  # Linearize testing images


# Generate T array from labels array (One-Hot Encoding)
def generateT(N, K, labelArr):
    DataT = np.zeros((N, K))  # Initialize DataT matrix to zeros of Size (NxK) 
    for i in range(0, labelArr.shape[0]):  # For loop to iterate through all labels
        DataT[i, labelArr[i]] = 1          # Create one-hot encoded row
    return DataT                           # Return one-hot encoded matrix

def normalizeAndGenerateDataMatrix(imageArr):
    imageArr = imageArr / 255  # Normalize pixels of images to range between 0 - 1
    return np.insert(imageArr, imageArr.shape[1], 1, axis=1)  # Insert Column of 1's at end of each image and Return Matrix

K = 10
NTest = imageTestArrLinearized.shape[0]       # Number of Testing Images
TTest = generateT(NTest, K, labelTestArr)     # Generate TTest Matrix from testing labels (one-hot encoding)
imageDataMatrixTest = normalizeAndGenerateDataMatrix(imageTestArrLinearized)      # Normalize and generate test data matrix from images array

totals = np.zeros((K,K))

yPred = np.dot(imageDataMatrixTest, W)

for i in range(len(yPred)):
    totals[np.argmax(yPred[i])][np.argmax(TTest[i])] += 1
np.set_printoptions(suppress=True)
print(totals)