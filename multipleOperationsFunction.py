__author__ = 'SoajanII'

import theano.tensor as T
from theano import function
from theano import pp
import numpy

x=T.dmatrix('x')
y=T.dmatrix('y')
absDiff=abs(x-y)
add2=x+y
f=function([x, y], [absDiff, add2])
a=numpy.array([[1, 2, 3, 4]])
b=numpy.array([[4, 27, 1, 8]])
print f(a, b)
