import numpy as np
import numpy.random as rnd

import logging

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FeatFactor
import pyfgraph.samples.logger as logger

def simple_variables(fg):
    V1 = fg.add(Variable, 'V1', domain=2)
    V2 = fg.add(Variable, 'V2', domain=2)
    V3 = fg.add(Variable, 'V3', domain=2)
    return V1, V2, V3

def make_simple_data(n):
    X = rnd.randn(n, 2)
    Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
    return { 'X': X, 'Y': Y }

def domain_variables(fg):
    V1 = fg.add(Variable, 'V1', domain=['!V1', 'V1'])
    V2 = fg.add(Variable, 'V2', domain=['!V2', 'V2'])
    V3 = fg.add(Variable, 'V3', domain=['!V3', 'V3'])
    return V1, V2, V3

def make_domain_data(n):
    X = rnd.randn(n, 2)
    Y = np.array([[ 'V1' if x[0]>=0 else '!V1',
                    'V2' if x[1]>=0 else '!V2',
                    'V3' if x[0]>=x[1] else '!V3' ] for x in X ], dtype=str)
    return { 'X': X, 'Y': Y }

def make_featgraph(vfun):
    fg = FactorGraph()
    V1, V2, V3 = vfun(fg)

    feats = fg.add(Feats, 'feats', nfeats = 2)

# Each factor has its own independent set of parameters
    F1 = fg.add(FeatFactor, 'F1', V1,           feats = feats)
    F2 = fg.add(FeatFactor, 'F2', V2,           feats = feats)
    F3 = fg.add(FeatFactor, 'F3', V3,           feats = feats)
    F_ = fg.add(FeatFactor, 'F_', (V1, V2, V3), feats = feats)

    fg.make()
    return fg

def simple_featgraph():
    return make_featgraph(vfun=simple_variables), make_simple_data

def domain_featgraph():
    return make_featgraph(vfun=domain_variables), make_domain_data

# if __name__ == '__main__':
#     fmt = '%(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
#     fmt = '%(asctime)s %(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
#     logging.basicConfig(filename='log.featgraph.log',
#                         filemode='w',
#                         format=fmt,
#                         level=logging.DEBUG)

#     fg, make_data = simple_featgraph()

#     data = make_data(n=20)
#     fg.train(data)

#     vit = fg.viterbi(data)
#     for x, y, v in zip(data['X'], data['Y'], vit):
#         print x, y, v
