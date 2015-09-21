__author__ = 'fhs20lted'

import theano.tensor as T
from theano import function
from theano import pp
import numpy

x=T.dmatrix('x')
y=T.dmatrix('y')
z=x+y
f=function([x,y],z)
testOut = f(numpy.array([[4,5,4],[1,3,5]]), numpy.array([[4,6,1],[2,37,1]]))
print testOut
print type(testOut)
print type(z)