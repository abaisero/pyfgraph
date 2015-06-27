import numpy as np
import numpy.random as rnd

from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, FunFactor

def simple_fungraph():
    def make_data(n):
        X = rnd.randn(n, 2)
        Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
        Y_kwlist = [ {'V1': y[0], 'V2': y[1], 'V3': y[2] } for y in Y ] 

        data = { 'X': X, 'Y': Y, 'Y_kwlist': Y_kwlist }
        return data

    fg = FactorGraph()

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    F1 = fg.add(FunFactor, 'F1', V1,           nfeats = 2)
    F2 = fg.add(FunFactor, 'F2', V2,           nfeats = 2)
    F3 = fg.add(FunFactor, 'F3', V3,           nfeats = 2)
    F_ = fg.add(FunFactor, 'F_', (V1, V2, V3), nfeats = 2)

    F1.set(lambda phi, V1: V1 * phi)
    F2.set(lambda phi, V2: V2 * phi)
    F3.set(lambda phi, V3: V3 * phi)
    F_.set(lambda phi, V1, V2, V3: (V1+V2+V3) * phi)

    fg.make(done=True)
    return fg, make_data