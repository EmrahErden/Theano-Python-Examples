__author__ = 'SoajanII'

def f(x):
    return x**2


class A(object):
    def __init__(self, m):
        self.m = m

    def f(self, x):
        return self.m*x**2


class B(A):
    def __init__(self, m):
        A.__init__(self, m)

    def f(self, x):
        return self.m*x


print f(2)
a = A(3)
print a.f(2)
b = B(3)
print b.f(2)