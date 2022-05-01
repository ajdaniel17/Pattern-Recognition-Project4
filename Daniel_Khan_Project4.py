import numpy as np
import glob
import cv2 as cv
import libsvm.svmutil as svm

path = input("Please enter the directory path containing test images:\n")

filenamesCElegans = glob.glob(path + '/*.png')
filenamesCElegans.sort()

def resizeAndConvertToGrayscale(filenames):
    images = [cv.imread(img) for img in filenames]

    scale_percent = 28
    width = int(101 * scale_percent / 100)
    height = int(101 * scale_percent / 100)
    dim = (width, height)

    imagesResized = [cv.resize(img, dim, interpolation=cv.INTER_AREA) for img in images]
    imagesGrayscale = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imagesResized]

    return imagesGrayscale

def printResults(filenames, yPred):
    print(" ___________________________________ ")
    print("|                    |              |")
    print("|    Image Name      |     Class    |")
    print("|                    |              |")
    print(" ___________________________________ ")

    K = 2

    totals = np.zeros((K))
    for i in range(len(filenames)):
        if yPred[i] == 1:
            print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),1))
            totals[1] += 1
        else:
            print("|  %s   |     %i      |" % (filenames[i].replace(path+"\\",""),0))
            totals[0] += 1

    print("Total Tallies:")
    for i in range(K):
        print("Class", int(i) , ":" , int(totals[i]))

imagesGrayscale = resizeAndConvertToGrayscale(filenamesCElegans)
imagesBlurred = [cv.GaussianBlur(img, (3,3), 0) for img in imagesGrayscale]

sobelx = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) for img in imagesBlurred]
sobely = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) for img in imagesBlurred]
sobelxy = [cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) for img in imagesBlurred]

edges = np.array([cv.Canny(image=img, threshold1=60, threshold2=140) for img in imagesBlurred])

imagesLinearized = edges.reshape(edges.shape[0], edges.shape[1] * edges.shape[2])

DataX = imagesLinearized / 255
DataX =  np.insert(DataX, DataX.shape[1], 1, axis=1)

m = svm.svm_load_model('libsvm.model')
labels,acc,vals = svm.svm_predict([],DataX,m,'-q')

printResults(filenamesCElegans, labels)