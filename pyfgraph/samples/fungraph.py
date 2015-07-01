import numpy as np
import numpy.random as rnd

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FunFactor
import pyfgraph.samples.logger as logger

def simple_fungraph():
    def make_data(n):
        X = rnd.randn(n, 2)
        Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
        Y_kwlist = [ {'V1': y[0], 'V2': y[1], 'V3': y[2] } for y in Y ] 

        data = { 'X': X, 'Y': Y, 'Y_kwlist': Y_kwlist }
        return data

    fg = FactorGraph()

    feats = fg.add(Feats, 'feats', nfeats = 2)

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    F1 = fg.add(FunFactor, 'F1', V1,           feats = feats)
    F2 = fg.add(FunFactor, 'F2', V2,           feats = feats)
    F3 = fg.add(FunFactor, 'F3', V3,           feats = feats)
    F_ = fg.add(FunFactor, 'F_', (V1, V2, V3), feats = feats)

    F1.fun = lambda feats, V1:          feats * V1
    F2.fun = lambda feats, V2:          feats * V2
    F3.fun = lambda feats, V3:          feats * V3
    F_.fun = lambda feats, V1, V2, V3:  feats * (V1+V2+V3)

    fg.make()
    return fg, make_data

if __name__ == '__main__':
    logger.setup_file_logger('log.fungraph.log')
    fg, make_data = simple_fungraph()

    data = make_data(n=20)
    fg.train(data)

    vit = fg.viterbi(data)
    for x, y, v in zip(data['X'], data['Y'], vit):
        print x, y, v

    print 'max:    {}'.format(fg.max())
    print 'argmax: {}'.format(fg.argmax())

