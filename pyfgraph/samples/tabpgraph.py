import numpy as np
import numpy.random as rnd

from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, TabPFactor
from pyfgraph.algo import message_passing
import logging

def simple_variables(fg):
    V1 = fg.add(Variable, 'V1', domain=2)
    V2 = fg.add(Variable, 'V2', domain=2)
    return V1, V2

def make_simple_data(n):
    X = rnd.randn(n, 2)
    Y = np.array([ [ x[0]>=.5, x[1]>=.5 ] for x in X ], dtype=int)
    return { 'X': [None]*n, 'Y': Y}

def make_tabpgraph(vfun):
    fg = FactorGraph()
    V1, V2 = vfun(fg)

    F1 = fg.add(TabPFactor, 'F1', V1          )
    F2 = fg.add(TabPFactor, 'F2', (V1, V2)    )

    fg.make()
    return fg

def simple_tabpgraph():
    return make_tabpgraph(vfun=simple_variables)

def domain_tabpgraph():
    return make_tabpgraph(vfun=domain_variables)

if __name__ == '__main__':
    fmt = '%(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    fmt = '%(asctime)s %(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    logging.basicConfig(filename='log.tabpgraph.log',
                        filemode='w',
                        format=fmt,
                        level=logging.DEBUG)

    fg = simple_tabpgraph()

    data = make_simple_data(10)

    fg.train(data)
    # message_passing(fg, 'max-product', 'sum-product')

    print 'argmax: {}'.format(fg.argmax())
    print 'max:    {}'.format(fg.max())

