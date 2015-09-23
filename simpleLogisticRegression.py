__author__ = 'SoajanII'

import theano.tensor as T
from theano import function
from theano import shared
from theano import pp
import numpy as np
import theano

nf = 3 #number of features
N = 1000 #number of examples
initRand = 0.01
numIterations = 100000
reg = 0.01

givenX = np.random.rand(N, nf)*2.-1 #input X values
givenY = ((givenX.sum(axis=1)>1).astype(int)).reshape(N,1) #input Y values
print 'given X : ' + str(givenX)
print 'given Y : ' + str(givenY)
print 'given X shape : ' + str(givenX.shape)
print 'given Y shape: ' + str(givenY.shape)
print " "
X = T.dmatrix('X')
Y = T.dmatrix('Y')

w = shared(np.random.rand(nf,1)*initRand, name='w')
b = shared(initRand, name='b')
print("Initial model:")
print 'w : ' + str((w.get_value()))
print 'b : ' + str((b.get_value()))

a = T.dot(X, w)+b
pred = 1 / (1 + T.exp(-a))
xent = -Y * T.log(pred) - (1-Y) * T.log(1-pred) #cross entropy
cost = xent.mean() + reg * (w ** 2).sum() #with regularization
gw, gb = T.grad(cost, [w, b])
train = function(inputs=[X,Y], outputs=[], updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

for i in range(numIterations):
    train(givenX, givenY)

print("final model:")
print 'w : ' + str((w.get_value()))
print 'b : ' + str((b.get_value()))
theano.printing.pydotprint(train, outfile="/home/fhs20lted/Desktop/Theano-Python-Examples/simpleLogisticRegression.png", var_with_name_simple=True)