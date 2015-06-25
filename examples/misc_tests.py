#! /usr/bin/python

import numpy as np

from pyfgraph.utils.decorators import kwargsdec
from pyfgraph.utils.iterators import proditer

def decorator():
    def f(a, b):
        return a + b

    g = kwargsdec(f)

    print f(a = 2, b = 1)
    print g(a = 2, b = 1, c = 4)

def iterator():
    for prodict in proditer(a = [1, 2 ,3], b = [2, 3, 4]):
        print prodict
        print prodict.values()

def slicing():
    n = 3
    a = np.zeros((n, n, n))
    for i in range(n):
        a[i, i, i] = 1

    idx = (2, 2)

    print a
    print idx
    print a[idx]

def tensor():
    a = np.array([1, 2])
    b = np.array([[1, 2, 3], [10, 20, 30]])

    print a.shape
    print b.shape

    print np.tensordot(a, b, 1)
    
if __name__ == '__main__':
    # decorator()
    # iterator()
    # slicing()
    tensor()

