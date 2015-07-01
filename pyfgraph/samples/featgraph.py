import numpy as np
import numpy.random as rnd

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FeatFactor
import pyfgraph.samples.logger as logger

def simple_featgraph():
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

# Each factor has its own independent set of parameters
    F1 = fg.add(FeatFactor, 'F1', V1,           feats = feats)
    F2 = fg.add(FeatFactor, 'F2', V2,           feats = feats)
    F3 = fg.add(FeatFactor, 'F3', V3,           feats = feats)
    F_ = fg.add(FeatFactor, 'F_', (V1, V2, V3), feats = feats)

    fg.make()
    return fg, make_data

if __name__ == '__main__':
    logger.setup_file_logger('log.featgraph.log')
    fg, make_data = simple_featgraph()

    data = make_data(n=20)
    fg.train(data)

    vit = fg.viterbi(data)
    for x, y, v in zip(data['X'], data['Y'], vit):
        print x, y, v

