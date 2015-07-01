import numpy as np
import numpy.random as rnd

from pyfgraph.parametric import Feats, Params
from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FeatFactor, FunFactor
import pyfgraph.samples.logger as logger

def simple_symbgraph():
    def make_data(n):
        X = rnd.randn(n, 5)
        Y = np.array([ [ x[0]>=0, x[1]>=0, x[2]>=x[3] and x[2]>=x[4] ] for x in X ], dtype=int)
        Y_kwlist = [ {'RIGHT': y[0], 'TOP': y[1], 'RED': y[2] } for y in Y ] 
        return { 'X':        X,
                 'Y':        Y,
                 'Y_kwlist': Y_kwlist }

    fg = FactorGraph()

    feats1 = fg.add(Feats, 'feats1', nfeats = 1)
    feats2 = fg.add(Feats, 'feats2', nfeats = 1)
    feats3 = fg.add(Feats, 'feats3', nfeats = 3)
    
    RIGHT = fg.add(Variable, 'RIGHT', arity = 2)
    TOP   = fg.add(Variable, 'TOP',   arity = 2)
    RED   = fg.add(Variable, 'RED',   arity = 2)
    V_ = (RIGHT, TOP, RED)

    F_RIGHT = fg.add(FeatFactor, 'F_RIGHT', RIGHT, feats = feats1)
    F_TOP   = fg.add(FeatFactor, 'F_TOP',   TOP,   feats = feats2)
    F_RED   = fg.add(FeatFactor, 'F_RED',   RED,   feats = feats3)
    F_      = fg.add(FunFactor,  'F_',      V_,    nfunfeats = 1)
    F_.fun  = lambda _values: np.array(sum(_values))

    fg.make()

    return fg, make_data

if __name__ == '__main__':
    logger.setup_file_logger('log.symbgraph.log')
    fg, make_data = simple_symbgraph()

    data = make_data(n=20)
    fg.train(data)

    vit = fg.viterbi(data)
    for x, y, v in zip(data['X'], data['Y'], vit):
        print x, y, v

