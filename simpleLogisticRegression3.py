__author__ = 'SoajanII'

from theano import tensor as T
from theano import function
from theano import shared
import numpy as np

N = 5000
nf = 3
initM = 0.01
reg = 0.
learningRate = 0.01
numIterations = 10000

inputX = np.random.rand(N, nf)*2.-1.
inputY = (inputX.sum(axis=1) > 0.).astype(int).reshape(N, 1)
treshold = 0.5

"""
print inputX
print inputY
print inputY.shape
"""

X = T.dmatrix('X')
Y = T.dmatrix('Y')
w = shared(np.random.rand(nf, 1)*initM, name='w')
b = shared(initM, name='b')
h = T.dot(X, w) + b
pred = 1/(1+T.exp(-h))
result = pred>treshold
crossEnt = -Y * T.log(pred) - (1-Y) * T.log(1-pred)
cost = crossEnt.mean()+reg*(w*w).sum()
err = abs(Y - result).mean()
gw, gb = T.grad(cost, [w, b])
train = function(inputs=[X, Y], outputs=err, updates=((w, w-learningRate*gw), (b, b-learningRate*gb)))

for i in range(numIterations):
    error = train(inputX, inputY)
    #print error

print error
print " "
print w.get_value()
print " "
print b.get_value()
print " "