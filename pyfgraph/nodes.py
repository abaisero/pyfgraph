import numpy as np
import numpy.random as rnd

# from params import Params
from utils.decorators import kwargsdec
from utils.iterators import proditer

import logging
logger = logging.getLogger(__name__)

class Node(object):
    def __init__(self, graph, vertex, name):
        self.graph = graph
        self.vertex = vertex
        self.name = name

    def __str__(self):
        return self.name

class Variable(Node):
    def __init__(self, graph, vertex, name, arity):
        super(Variable, self).__init__(graph, vertex, name)
        self.arity = arity
        self.domain = np.arange(arity)
        self.value = None

class Factor(Node):
    def __init__(self, graph, vertex, name, variables):
        super(Factor, self).__init__(graph, vertex, name)
        self.variables = variables if isinstance(variables, tuple) else (variables,)
        self.nvariables = len(self.variables)
        self.arity = tuple( v.arity for v in self.variables )
        self.table = None

    def value(self, idx = None):
        if self.table is None:
            raise Exception
        if idx is None:
            idx = tuple( v.value for v in self.variables )
        value = self.table[idx]
        logger.debug('%s.value(): %s', self.name, value)
        return self.table[idx]

    def set(self, *args, **kwargs):
        logger.error('Factor.set() is an interface and is not implemented yet.')
        raise NotImplementedError

    def make(self, feats, params):
        logger.error('Factor.make() is an interface and is not implemented yet.')
        raise NotImplementedError

    def make_table(self, x_kw, y_kw=None):
        logger.error('Factor.make_table() is an interface and is not implemented yet.')
        raise NotImplementedError

    def gradient(self):
        logger.error('Factor.gradent() is an interface and is not implemented yet.')
        raise NotImplementedError

class TabFactor(Factor):
    def __init__(self, *args, **kwargs):
        super(TabFactor, self).__init__(*args, **kwargs)
        self._table = None

    @property
    def table(self):
        return self._table

    @table.setter
    def table(self, value):
        if isinstance(value, (int, long, float)):
            self._table.fill(value)
        elif isinstance(value, list):
            self._table = np.array(value)
        elif isinstance(value, np.ndarray):
            self._table = value
        else:
            NotImplementedError('{}.table.setter() failed with type(table) = {}'.format(self.name, type(value)))

    # def set(self, table):
    #     if isinstance(table, (int ,long, float)):
    #         self.table.fill(table)
    #     elif isinstance(table, list):
    #         self.table = np.array(table)
    #     elif isinstance(table, np.ndarray):
    #         self.table = table
    #     else:
    #         NotImplementedError('setTable failed with type(table) = {}'.format(type(table)))

    def make(self, feats, params):
        pass

    def make_table(self, x_kw = None, y_kw = None):
        pass

class ParamFactor(Factor):
    fslices = {}
    nfeats = 0

    pslices = {}
    nparams = 0

    def __init__(self, graph, vertex, name, variables):
        super(ParamFactor, self).__init__(graph, vertex, name, variables)
        self._fslice = None
        self._pslice = None

    @property
    def fslice(self):
        return self._fslice

    @fslice.setter
    def fslice(self, value):
        if value is None:
            self._fslice = slice(None)
            self.nfeats = self.graph.fdesc._nfeats
        elif isinstance(value, slice):
            self._fslice = value
            self.nfeats = self.graph.fdesc._fdict[value]

    @property
    def pslice(self):
        return self._pslice

    @pslice.setter
    def pslice(self, value):
        if value is None or value not in ParamFactor.pslices:
            self._pslice = slice(ParamFactor.nparams, ParamFactor.nparams+self.nparams)
            ParamFactor.nparams += self.nparams
        else:
            self._pslice = ParamFactor.pslices[value]

        if value is not None and value not in ParamFactor.pslices:
            ParamFactor.pslices[value] = self._pslice

from operator import mul

class FeatFactor(ParamFactor):
    def __init__(self, graph, vertex, name, variables, fid = None, pid = None):
        super(FeatFactor, self).__init__(graph, vertex, name, variables)

# TODO get the fslice related wit this fid

        self.fslice = fid
        print self.nfeats
        self.nparams = reduce(mul, self.arity) * self.nfeats
        self.pslice = pid

    def make(self, feats, params):
        self.feats = feats[self.fslice]

        self.params = params[self.pslice]
        self.params_tab = self.params.view()
        self.params_tab.shape = self.arity + (-1,)
        
        logger.debug('%s.fslice:           %s', self.name, self.fslice)
        logger.debug('%s.feats.shape:      %s', self.name, self.feats.shape)
        logger.debug('%s.pslice:           %s', self.name, self.pslice)
        logger.debug('%s.params.shape:     %s', self.name, self.params.shape)
        logger.debug('%s.params_tab.shape: %s', self.name, self.params_tab.shape)

    def make_table(self, x_kw=None, y_kw=None):
# set/make table
        self.nl_table = np.dot(self.params_tab, self.feats)
        self.table = np.exp(-self.nl_table)

    def gradient(self):
        idx = tuple( v.value for v in self.variables )
        pr = self.graph.pr(self.vertex)

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
    def __init__(self, graph, vertex, name, variables, fid = None, pid = None):
        super(FunFactor, self).__init__(graph, vertex, name, variables)

        self.fslice = fid
        self.nparams = self.nfeats
        self.pslice = pid

        self._fun = None

    @property
    def fun(self):
        return self._fun

    @fun.setter
    def fun(self, value):
        if callable(value):
            self._fun = kwargsdec(value)
        else:
            raise Exception('{}.fun.setter() requires a callable as argument.'.format(self.name))

    # def set(self, fun):
    #     self.fun = kwargsdec(fun)

    def make(self, feats, params):
        self.params = params[self.pslice]
        
        logger.debug('%s.fslice:           %s', self.name, self.fslice)
        logger.debug('%s.feats.shape:      %s', self.name, self.feats.shape)
        logger.debug('%s.pslice:           %s', self.name, self.pslice)
        logger.debug('%s.params.shape:     %s', self.name, self.params.shape)
        logger.debug('%s.params_tab.shape: %s', self.name, self.params_tab.shape)

    def make_table(self, x_kw=None, y_kw=None):
# make/set features
        vdict = self.graph.vdict
# TODO fix this
        self.feats = np.array([ self.fun(fg = self.graph, **dict(x_kw, **prodict)) for prodict in proditer(**vdict) ])
        self.feats.shape = self.arity + (-1,)
# TODO unnecessary?
        # self.nfeats = self.feats.shape[-1]

# make/set table
        self.nl_table = np.dot(self.feats, self.params)
        self.table = np.exp(-self.nl_table)
        
    def gradient(self):
        idx = tuple( v.value for v in self.variables)
        pr = self.graph.pr(self.vertex)

        g = np.zeros(ParamFactor.nparams)
        g[self.pslice] = self.feats[idx] - np.tensordot(pr, self.feats, self.nvariables)
        return g

