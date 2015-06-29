import numpy as np
import numpy.random as rnd

from pyfgraph.fgraph import FactorGraph, Feats
from pyfgraph.nodes import Variable, FeatFactor, FunFactor

def simple_symbgraph():
    def make_data(n):
        X = rnd.randn(n, 2)
        Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
        Y_kwlist = [ {'V1': y[0], 'V2': y[1], 'V3': y[2] } for y in Y ] 

        data = { 'X': X, 'Y': Y, 'Y_kwlist': Y_kwlist }
        return data

    fg = FactorGraph()

    phi1 = fg.add(Feature, 'phi1', nfeats=2)
    phi2 = fg.add(Feature, 'phi2', nfeats=2)
    phi3 = fg.add(Feature, 'phi3', nfeats=2)
    phi = fg.feats

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    F1 = fg.add(FeatFactor, 'F1', V1,           feats = phi1)
    F2 = fg.add(FeatFactor, 'F2', V2,           feats = phi2)
    F3 = fg.add(FeatFactor, 'F3', V3,           feats = phi3)
    F_ = fg.add(FunFactor,  'F_', (V1, V2, V3), feats = phi)

# the following variables can be accessed in this function:
#   * V1, V2, V3
#   * phi_all, phi
#   * fg
    F_.fun = lambda fg, feats: sum(fg.values) * feats

    fg.make(done=True)
    return fg, make_data
