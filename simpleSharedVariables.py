__author__ = 'SoajanII'

import theano.tensor as T
from theano import function
from theano import pp
from theano import shared

sharedVar = shared(0)
x=T.iscalar('x')
accumulator=function([x], updates=[(sharedVar, sharedVar+x)])
print sharedVar.get_value()
accumulator(5)
print sharedVar.get_value()
sharedVar.set_value(33)
print sharedVar.get_value()
accumulator(-22)
print sharedVar.get_value()