from collections import namedtuple

import numpy as np
import numpy.random as rnd

import logging

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FeatFactor
import pyfgraph.samples.logger as logger

def make_data(n):
    X = rnd.randn(n, 2)
    X2 = np.empty((n, 2, 2, 2))
    for i in range(n):
        X2[i, :] = X[i]

    Y = np.array([ [ x[0]>=0, x[1]>=0 ] for x in X ], dtype=int)

    Feats = namedtuple('Feats', 'feats')
    feats = [ Feats(feats=x) for x in X2 ]

    Values = namedtuple('Values', 'V1, V2')
    values = [ Values(V1=y[0], V2=y[1]) for y in Y ]

    IValues = namedtuple('IValues', 'V1, V2')
    ivalues = [ IValues(V1=y[0], V2=y[1]) for y in Y ]

    Data = namedtuple('Data', 'feats, values, ivalues')
    return [ Data(feats=f, values=v, ivalues=iv) for f, v, iv in zip(feats, values, ivalues) ]

def simple_featgraph():
    fg = FactorGraph()

    V1 = fg.add(Variable, 'V1', domain=2)
    V2 = fg.add(Variable, 'V2', domain=2)

    feats = fg.add(Feats, 'feats', nfeats = 2)
    F1 = fg.add(FeatFactor, 'F1', (V1, V2), feats = feats)

    fg.make()
    return fg, make_data

if __name__ == '__main__':
    fmt = '%(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    fmt = '%(asctime)s %(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    logging.basicConfig(filename='log.newfeatgraph.log',
                        filemode='w',
                        format=fmt,
                        level=logging.DEBUG)

    fg, make_data = simple_featgraph()

    data = make_data(n=20)
    fg.train(data)

    print fg.log_pr(fg.factors[0])
    print fg.log_pr(fg.variables[0])
    print fg.log_pr(fg.variables[1])

    # vit = fg.viterbi(data)
    # for d, v in zip(data, vit):
    #     print d.feats.feats, d.values, v
