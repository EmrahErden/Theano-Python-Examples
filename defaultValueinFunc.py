__author__ = 'SoajanII'

import theano.tensor as T
from theano import function
from theano import pp
from theano import Param
import numpy

x = T.dmatrix('x')
y = T.dscalar('y')
out = x ** y
f=function([x, Param(y, default = 2)], out)
print f(numpy.array([[1,2,3]]))
print f(numpy.array([[1,2,3]]),3)