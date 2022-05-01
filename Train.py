import numpy as np
import libsvm.svmutil as svm
import math
import time

data = np.load('DataX.npz')
DataX = data['DataX']

data = np.load('DataT.npz')
DataT = data['DataT']

N, D = DataX.shape
DataT = DataT.reshape(N)

# Divide DataX into training and testing sets
DataXTrain = DataX[:math.ceil(DataX.shape[0] * 0.8)]
DataXTest = DataX[math.ceil(DataX.shape[0] * 0.8):]

# Divide DataT into training and testing sets
DataTTrain = DataT[:math.ceil(DataT.shape[0] * 0.8)]
DataTTest = DataT[math.ceil(DataX.shape[0] * 0.8):]

options = '-t 2 -h 0'
start_time = time.time()
model = svm.svm_train(DataTTrain.astype(int).tolist(), DataXTrain.tolist(), options)
end_time = time.time()
print("Total Training Time :", end_time-start_time)

start_time = time.time()
labels,acc,vals = svm.svm_predict(DataTTest,DataXTest,model)
end_time = time.time()
print("Total Testing Time :", end_time-start_time)

svm.svm_save_model('libsvm.model', model)

#Confusion Matrix

ConfuseMatrix = np.zeros((2,2))

for i in range(len(DataTTest)):
    if DataTTest[i] == -1 :
        if labels[i] == -1:
            ConfuseMatrix[0][0] += 1
        else:
            ConfuseMatrix[1][0] += 1
    else:
        if labels[i] == -1:
            ConfuseMatrix[0][1] += 1
        else:
            ConfuseMatrix[1][1] += 1
print(len(DataTTest))
print(ConfuseMatrix)