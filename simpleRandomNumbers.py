__author__ = 'SoajanII'

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=2222)
rn_t1 = srng.normal((2,2))
rn_t2 = srng.normal((2,2))

f=function([], rn_t1)
g=function([], rn_t2, no_default_updates=True)
h=function([], rn_t1+rn_t1-2*rn_t1)
print f()
print f()
print g()
print g()
print h()
print h()


