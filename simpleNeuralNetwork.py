__author__ = 'SoajanII'

import numpy as np

nf = 4 # number of features
no = 3 # number of outputs
N = 150 #total number of examples
fracTrain = 0.1 #fraction of training examples to all examples
fileName = 'iris.txt'

ls1 = 20
ls2 = 20

#initializing the test and train matrices
nTrain = int(N*fracTrain)
nTest = N - nTrain
Xtrain=np.zeros(shape=(nTrain, nf)) #X declaration
Ytrain=np.zeros(shape=(nTrain, no)) #Y declaration
Xtest=np.zeros(shape=(nTest, nf)) #X declaration
Ytest=np.zeros(shape=(nTest, no)) #Y declaration

#reading the file and writing to the big matrix
X_and_Y=np.zeros(shape=(N, nf+no)) #X declaration
file = open(fileName, 'r')
line = file.readline()
lineCounter = 0

while(lineCounter<N):
    datas = line.split()
    for i in range(0,nf):
        X_and_Y[lineCounter, i] = datas[i]
    if(datas[nf]=='I.setosa'):
        X_and_Y[lineCounter, nf:] = [1, 0, 0]
    elif(datas[nf]=='I.versicolor'):
        X_and_Y[lineCounter, nf:] = [0, 1, 0]
    elif(datas[nf]=='I.virginica'):
        X_and_Y[lineCounter, nf:] = [0, 0, 1]
    lineCounter=lineCounter+1
    line = file.readline()
#shuffling the big matrix in the end to for Xtrain, Ytrain, Xtest, Ytest
np.random.shuffle(X_and_Y)

Xtrain = X_and_Y[0:nTrain, 0:nf]
Ytrain = X_and_Y[0:nTrain, nf:]
Xtest = X_and_Y[nTrain:, 0:nf]
Ytest = X_and_Y[nTrain:, nf:]

#stop, train time!!