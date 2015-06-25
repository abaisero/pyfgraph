#! /usr/bin/python

import numpy as np
import numpy.random as rnd

import pyfgraph
from pyfgraph.fgraph import FactorGraph
from pyfgraph.params import Params
from pyfgraph.nodes import Variable, Factor, FeatFactor
from pyfgraph.algo import message_passing

import os, logging

def log_setup():
    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logging.basicConfig(
            filename=os.path.basename(__file__) + '.log',
            filemode='w',
            format=fmt,
            level=logging.INFO)

    logger = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def example_indep_params():
    log_setup()

    fg = FactorGraph()

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    P1 = Params(n=4)
    P2 = Params(n=4)
    P3 = Params(n=4)
    P_ = Params(n=16)

    F1 = fg.add(FeatFactor, 'F1', V1,           P1)
    F2 = fg.add(FeatFactor, 'F2', V2,           P2)
    F3 = fg.add(FeatFactor, 'F3', V3,           P3)
    F_ = fg.add(FeatFactor, 'F_', (V1, V2, V3), P_)

    fg.make(done=True)
    return fg

def example_dep_params():
    fg = FactorGraph()

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    PV = Params(n=4)
    P_ = Params(n=16)

    # Feat = Feats(n=2)

# Factors 1, 2 and 3 share the same parameters
    # F1 = fg.add(FeatFactor, 'F1', V1,           PV, Feat)
    # F2 = fg.add(FeatFactor, 'F2', V2,           PV, Feat)
    # F3 = fg.add(FeatFactor, 'F3', V3,           PV, Feat)
    # F_ = fg.add(FeatFactor, 'F_', (V1, V2, V3), P_, Feat)
    F1 = fg.add(FeatFactor, 'F1', V1,           PV)
    F2 = fg.add(FeatFactor, 'F2', V2,           PV)
    F3 = fg.add(FeatFactor, 'F3', V3,           PV)
    F_ = fg.add(FeatFactor, 'F_', (V1, V2, V3), P_)

    fg.make(done=True)
    return fg

def make_data(n):
    X = rnd.randn(n, 2)
    Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)
    Y_kwlist = [ {'V1': y[0], 'V2': y[1], 'V3': y[2] } for y in Y ] 

    data = { 'X': X, 'Y': Y, 'Y_kwlist': Y_kwlist }
    return data

if __name__ == '__main__':
    rnd.seed(1)

# Notice that, given the same training and testing data, the model with
# independent parameters achieves better performance.
    # fg = example_indep_params()
    fg = example_dep_params()

    data = make_data(10)
    print 'Training begin.'
    fg.train(data)
    print 'Training done.'

    print 'Results (on training data):'
    nll = fg.nll(data=data)
    vit = fg.viterbi(data=data)
    for x, y, v, l in zip(data['X'], data['Y'], vit, nll):
        print x, y, v, l

    data = make_data(10)
    print 'Results (on test data):'
    nll = fg.nll(data=data)
    vit = fg.viterbi(data=data)
    for x, y, v, l in zip(data['X'], data['Y'], vit, nll):
        print x, y, v, l

