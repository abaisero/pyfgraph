import numpy as np
import numpy.random as rnd

from parametric import Params
from utils.decorators import kwargsdec
from utils.iterators import proditer

from operator import mul

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
    def __init__(self, graph, vertex, name, variables, feats=None, params=None):
        super(Factor, self).__init__(graph, vertex, name)
        self.variables = variables if isinstance(variables, tuple) else (variables,)
        self.feats = feats
        self.params = params

        self.arity = tuple( v.arity for v in self.variables )
        self.table = None

    @property
    def nvariables(self):
        return len(self.variables)

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

    def make_table(self):
        logger.error('Factor.make_table() is an interface and is not implemented yet.')
        raise NotImplementedError

    def gradient(self):
        logger.error('Factor.gradent() is an interface and is not implemented yet.')
        raise NotImplementedError

class TabFactor(Factor):
    def __init__(self, graph, vertex, name, variables):
        super(TabFactor, self).__init__(graph, vertex, name, variables)
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

    def make(self):
        pass

    def make_table(self):
        pass

class FeatFactor(Factor):
    def __init__(self, graph, vertex, name, variables, feats = None, params = None):
        super(FeatFactor, self).__init__(graph, vertex, name, variables, feats, params)
        if self.params is None:
            self.params = self.graph.add(Params, '_{}.param'.format(name), nparams=reduce(mul, self.arity)*self.feats.nfeats)

# TODO if no params is given, create a new parameter which matchec this type of Factor
        # self.nparams = reduce(mul, self.arity) * self.feats.nfeats

    def make(self):
        self.params_tab = self.params.params.view()
        self.params_tab.shape = self.arity + (-1,)

    def make_table(self):
# set/make table

        # print 'feats:', self.feats.feats
        # print 'params_tab', self.params_tab
        self.nl_table = np.dot(self.params_tab, self.feats.feats)
        # print 'nl_table', self.nl_table
        self.table = np.exp(-self.nl_table)

    def gradient(self):
        idx = tuple( v.value for v in self.variables )
        pr = self.graph.pr(self.vertex)

        ttable = np.zeros(self.arity)
        ttable[idx] = 1

        # logger.debug('%s.gradient():', self.name)
        # logger.debug(' * ttable: %s', ttable)
        # logger.debug(' * pr:     %s', pr)
        # logger.debug(' * first:  %s', np.kron(ttable, self.feats.feats))
        # logger.debug(' * second: %s', np.kron(pr, self.feats.feats))

        g = np.zeros(Params.nparams)
        # g[self.pslice] = (np.kron(ttable, self.feats) - np.kron(pr, self.feats)).ravel()
        g[self.params.pslice] = np.kron(ttable-pr, self.feats.feats).ravel()
        return g

class FunFactor(Factor):
    def __init__(self, graph, vertex, name, variables, feats = None, params = None, nfunfeats = None):
        super(FunFactor, self).__init__(graph, vertex, name, variables, feats, params)

        if self.feats is None and nfunfeats is None:
            raise Exception('FunFactor.__init__() requires either `feats` or `nfunfeats` to be specified')
        if self.feats is not None and nfunfeats is not None and self.feats.nfeats != nfunfeats:
            raise Exception('FunFactor.__init__() receives incompatible `feats` and `nfunfeats` parameters')

        self.params = self.graph.add(Params, '_{}.param'.format(name), nparams=nfunfeats or self.feats.nfeats)
        # if nfunfeats is None:
        #     nfunfeats = self.feats.nfeats
        # self.params = self.graph.add(Params, '_{}.param'.format(name), nparams=nfunfeats)
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

    def make(self):
        pass

    def make_table(self):
# make/set features
        kwargs = { '_graph': self.graph }
        kwargs.update({ feat.name: feat.feats for feat in self.graph.features })
# update with variable values
        # print kwargs
        vdict = self.graph.vdict(self.variables)
        self.funfeats = np.array([ self.fun(_values = prodict.values(), **dict(kwargs, **prodict)) for prodict in proditer(**vdict) ])
        self.funfeats.shape = self.arity + (-1,)

# make/set table
        self.nl_table = np.dot(self.funfeats, self.params.params)
        self.table = np.exp(-self.nl_table)
        
    def gradient(self):
        idx = tuple( v.value for v in self.variables)
        pr = self.graph.pr(self.vertex)

        g = np.zeros(Params.nparams)
        g[self.params.pslice] = self.funfeats[idx] - np.tensordot(pr, self.funfeats, self.nvariables)
        return g

