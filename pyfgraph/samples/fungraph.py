import numpy as np
import numpy.random as rnd

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FunFactor

import logging

def simple_fungraph():
    def make_data(n):
        X = rnd.randn(n, 2)
        Y = np.array([ [ 'RIGHT' if x[0]>=0    else 'LEFT',
                         'TOP'   if x[1]>=0    else 'BOTTOM',
                         'BL'    if x[0]>=x[1] else 'TR' ] for x in X ])
        # Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=bool)
        # Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
        return { 'X': X, 'Y': Y }

    fg = FactorGraph()

    feats = fg.add(Feats, 'feats', nfeats = 2)

    # V1 = fg.add(Variable, 'V1', domain=2)
    # V2 = fg.add(Variable, 'V2', domain=2)
    # V3 = fg.add(Variable, 'V3', domain=2)
    # V1 = fg.add(Variable, 'V1', domain=[False, True])
    # V2 = fg.add(Variable, 'V2', domain=[False, True])
    # V3 = fg.add(Variable, 'V3', domain=[False, True])
    V1 = fg.add(Variable, 'V1', domain=['RIGHT', 'LEFT'])
    V2 = fg.add(Variable, 'V2', domain=['TOP', 'BOTTOM'])
    V3 = fg.add(Variable, 'V3', domain=['TR', 'BL'])

    F1 = fg.add(FunFactor, 'F1', V1,           feats = feats)
    F2 = fg.add(FunFactor, 'F2', V2,           feats = feats)
    F3 = fg.add(FunFactor, 'F3', V3,           feats = feats)
    F_ = fg.add(FunFactor, 'F_', (V1, V2, V3), feats = feats)

    F1.fun = lambda feats, V1:          feats * (V1 == 'RIGHT')
    F2.fun = lambda feats, V2:          feats * (V2 == '')
    F3.fun = lambda feats, V3:          feats * V3
    F_.fun = lambda feats, V1, V2, V3:  feats * (V1+V2+V3)

    F1.fun = lambda feats, V1:          feats * (V1 == 'RIGHT')
    F2.fun = lambda feats, V2:          feats * (V2 == '')
    F3.fun = lambda feats, V3:          feats * V3
    F_.fun = lambda feats, V1, V2, V3:  feats * (V1+V2+V3)

    # TODO alternatively: lambda s: s.feats * (s.V1 == 'RIGHT')

    fg.make()
    return fg, make_data

if __name__ == '__main__':
    fmt = '%(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    fmt = '%(asctime)s %(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    logging.basicConfig(filename='log.featgraph.log',
                        filemode='w',
                        format=fmt,
                        level=logging.DEBUG)

    fg, make_data = simple_fungraph()

    data = make_data(n=20)
    fg.train(data)

    vit = fg.viterbi(data)
    for x, y, v in zip(data['X'], data['Y'], vit):
        print x, y, v

    print 'max:    {}'.format(fg.max())
    print 'argmax: {}'.format(fg.argmax())

