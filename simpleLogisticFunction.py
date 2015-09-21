__author__ = 'fhs20lted'

from theano import function
from theano import pp
import theano.tensor as T
import numpy

x=T.dmatrix('x')
s=1./(1+T.exp(-x))
logistic=function([x], s)
output = logistic(numpy.array([[-400, 500, 1, 0]]))
print output
