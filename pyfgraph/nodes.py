import numpy as np
import numpy.random as rnd

# from params import Params
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
    def __init__(self, vertex, name, variables):
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

    def make_table(self, phi = None, y_kw = None):
        logger.error('Factor.make_table() is an interface and is not implemented yet.')
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

    def make_table(self, phi = None, y_kw = None):
        pass

class ParamFactor(Factor):
    pslices = {}
    nparams = 0
    def __init__(self, vertex, name, variables):
        super(ParamFactor, self).__init__(vertex, name, variables)
        self.pslice = None

    def set_pslice(self):
        if self.pid is None or self.pid not in ParamFactor.pslices:
            self.pslice = slice(ParamFactor.nparams, ParamFactor.nparams+self.nparams)
            ParamFactor.nparams += self.nparams
        else:
            self.pslice = ParamFactor.pslices[self.pid]

        if self.pid is not None and self.pid not in ParamFactor.pslices:
            ParamFactor.pslices[self.pid] = self.pslice

from operator import mul

class FeatFactor(ParamFactor):
    def __init__(self, vertex, name, variables, nfeats, pid = None):
        super(FeatFactor, self).__init__(vertex, name, variables)

        self.nfeats = nfeats
        self.nparams = reduce(mul, self.arity) * nfeats
        self.pid = pid
        self.set_pslice()

    def make_params(self, params):
        self.params = params[self.pslice]
        self.params_tab = self.params.view()
        self.params_tab.shape = self.arity + (-1,)

    def make_table(self, phi, y_kw = None):
# set/make features
        self.feats = phi
# unnecessary. But I could use it to check stuff
        # self.nfeats = len(self.feats)

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

        g = np.zeros(ParamFactor.nparams)
        # g[self.pslice] = (np.kron(ttable, self.feats) - np.kron(pr, self.feats)).ravel()
        g[self.pslice] = np.kron(ttable-pr, self.feats).ravel()
        return g

class FunFactor(ParamFactor):
    def __init__(self, vertex, name, variables, nfeats, pid = None):
        super(FunFactor, self).__init__(vertex, name, variables)

        self.nfeats = nfeats
        self.nparams = nfeats
        self.pid = pid
        self.set_pslice()

        self.fun = None

    def set(self, fun):
        self.fun = kwargsdec(fun)

    def make_params(self, params):
        self.params = params[self.pslice]

    def make_table(self, phi, y_kw = None):
# make/set features
        iterators = { v.name: v.domain for v in self.variables }
        self.feats = np.array([ self.fun(phi=phi, variables=prodict.values(), **prodict) for prodict in proditer(**iterators) ])
        self.feats.shape = self.arity + (-1,)
# TODO unnecessary?
        # self.nfeats = self.feats.shape[-1]

# make/set table
        self.nl_table = np.dot(self.feats, self.params)
        self.table = np.exp(-self.nl_table)
        
    def gradient(self):
        idx = tuple( v.value for v in self.variables)
        pr = self.graph.pr(self.vertex, with_l1_norm=False)

        g = np.zeros(ParamFactor.nparams)
        g[self.pslice] = self.feats[idx] - np.tensordot(pr, self.feats, self.nvariables)
        return g

