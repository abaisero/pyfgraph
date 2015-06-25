import numpy as np
import numpy.random as rnd

from params import Params
from utils.decorators import kwargsdec
from utils.iterators import proditer

import logging
logger = logging.getLogger(__name__)

class Node(object):
    def __init__(self, vertex, name):
        self.vertex = vertex
        self.name = name

    def __str__(self):
        return self.name

class Variable(Node):
    def __init__(self, vertex, name, arity):
        super(Variable, self).__init__(vertex, name)
        self.arity = arity
        self.domain = np.arange(arity)
        self.value = None

class Factor(Node):
    def __init__(self, vertex, name, variables, params=None, feats=None):
        super(Factor, self).__init__(vertex, name)
        self.variables = variables if isinstance(variables, tuple) else (variables,)
        self.nvariables = len(self.variables)
        self.arity = tuple( v.arity for v in self.variables )
        self.table = None

    def value(self, idx = None):
        if self.table is None:
            raise Exception
        if idx is None:
            idx = tuple( v.value for v in self.variables )
        return self.table[idx]

    def set(self, *args, **kwargs):
        logger.error('Factor.set() is an interface and is not implemented yet.')
        raise NotImplementedError

    def make(self, phi = None, y_kw = None):
        logger.error('Factor.make() is an interface and is not implemented yet.')
        raise NotImplementedError

    def gradient(self):
        logger.error('Factor.gradent() is an interface and is not implemented yet.')
        raise NotImplementedError

class TabFactor(Factor):
    def __init__(self, *args, **kwargs):
        super(TabFactor, self).__init__(*args, **kwargs)

    def set(self, table):
        if isinstance(table, (int ,long, float)):
            self.table.fill(table)
        elif isinstance(table, list):
            self.table = np.array(table)
        elif isinstance(table, np.ndarray):
            self.table = table
        else:
            NotImplementedError('setTable failed with type(table) = {}'.format(type(table)))

    def make(self, phi = None, y_kw = None):
        pass

class FeatFactor(Factor):
    def __init__(self, vertex, name, variables, params = None, feats = None):
        super(FeatFactor, self).__init__(vertex, name, variables)
        self.nfeats = 0
        self.feats = None
        self.pshape = self.arity + (-1,)

        self.params = params
        # self.feats = feats

    def make(self, phi, y_kw = None):
# set/make features
        self.feats = phi
        self.nfeats = len(self.feats)

# set/make table
        self.nl_table = np.dot(self.params_tab, self.feats)
        self.table = np.exp(-self.nl_table)

    def gradient(self):
        idx = tuple( v.value for v in self.variables )
        pr = self.graph.pr(self.vertex, with_l1_norm=False)

        ttable = np.zeros(self.arity)
        ttable[idx] = 1

        logger.debug('%s.gradient():', self.name)
        logger.debug(' * ttable: %s', ttable)
        logger.debug(' * pr:     %s', pr)
        logger.debug(' * first:  %s', np.kron(ttable, self.feats))
        logger.debug(' * second: %s', np.kron(pr, self.feats))

        g = np.zeros(Params.tot_nparams)
        # g[self.pslice] = (np.kron(ttable, self.feats) - np.kron(pr, self.feats)).ravel()
        g[self.pslice] = np.kron(ttable-pr, self.feats).ravel()
        return g

class FunFactor(Factor):
    def __init__(self, vertex, name, variables, params = None):
        super(FunFactor, self).__init__(vertex, name, variables)
        self.params = params
        self.fun = None
        self.pshape = self.arity + (-1,)

    def set(self, fun):
        self.fun = kwargsdec(fun)

    def make(self, phi, y_kw = None):
# make/set features
        iterators = { v.name: v.domain for v in self.variables }
        self.feats = np.array([ self.fun(phi=phi, **prodict) for prodict in proditer(**iterators) ])
        self.feats.shape = self.arity + (-1,)
        self.nfeats = self.feats.shape[-1]

# make/set table
        self.nl_table = np.dot(self.feats, self.params)
        self.table = np.exp(-self.nl_table)
        
    def gradient(self):
        idx = tuple( v.value for v in self.variables)
        pr = self.graph.pr(self.vertex, with_l1_norm=False)

        g = np.zeros(Params.tot_nparams)
        g[self.pslice] = self.feats[idx] - np.tensordot(pr, self.feats, self.nvariables)
        return g

