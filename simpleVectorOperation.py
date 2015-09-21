__author__ = 'fhs20lted'

"""
import theano
a = theano.tensor.vector() # declare variable
out = a + a ** 10               # build symbolic expression
f = theano.function([a], out)   # compile function
print(f([0, 1, 2]))

#[    0.     2.  1026.]
#Modify and execute this code to compute this expression: a ** 2 + b ** 2 + 2 * a * b.
"""

import theano.tensor as T
from theano import function

a=T.vector('a')
b=T.vector('b')
out=a**2+b**2+2*a*b
f=function([a,b],out)
print f([0,1,2],[1,2,2])

