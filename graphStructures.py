__author__ = 'SoajanII'

import theano.tensor as T
from theano import function
from theano import pp

x=T.dmatrix('x')
y=T.dmatrix('y')
z=x*y+x**2

print type(x.owner)
print type(z.owner)
print z.owner.op.name
print z.owner.inputs
