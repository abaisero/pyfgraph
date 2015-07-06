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
    def __init__(self, graph, vertex, name, domain):
        super(Variable, self).__init__(graph, vertex, name)
        if isinstance(domain, int):
            domain = range(domain)
        self.domain = domain
        self.arity = len(domain)
        self.idomain = np.arange(self.arity)

        self.v2iv = dict(zip(self.domain, self.idomain))
        self.iv2v = dict(zip(self.idomain, self.domain))

        logger.info('Variable %s', self.name)
        logger.info(' - domain: %s', self.domain)
        logger.info(' - idomain: %s', self.idomain)
        logger.info(' - v2iv: %s', self.v2iv)
        logger.info(' - iv2v: %s', self.iv2v)

        self._value = None
        self._ivalue = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is None:
            self._value = None
            self._ivalue = None
        else:
            self._value = value
            self._ivalue = self.v2iv[value]

    @property
    def ivalue(self):
        return self._ivalue

    @ivalue.setter
    def ivalue(self, ivalue):
        if ivalue is None:
            self._ivalue = None
            self._value = None
        else:
            print 'setting ivalue {} (value {})'.format(ivalue, self.iv2v[ivalue])
            self._ivalue = ivalue
            self._value = self.iv2v[ivalue]

class Factor(Node):
    def __init__(self, graph, vertex, name, variables, feats=None, params=None):
        super(Factor, self).__init__(graph, vertex, name)
        self.variables = variables if isinstance(variables, tuple) else (variables,)
        self.feats = feats
        self.params = params

        self.arity = tuple( v.arity for v in self.variables )
        self.table = None
        self.nl_table = None

    @property
    def nvariables(self):
        return len(self.variables)

    # def value(self, idx = None):
    #     if self.table is None:
    #         raise Exception
    #     if idx is None:
    #         idx = tuple( v.ivalue for v in self.variables )

    #     value = self.table[idx]
    #     return value

    def nl_value(self, idx = None):
        if self.nl_table is None:
            raise Exception
        if idx is None:
            idx = tuple( v.ivalue for v in self.variables )

        nl_value = self.nl_table[idx]
        return nl_value

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
            if self._table is None:
                self._table = np.empty(self.arity)
            self._table.fill(value)
        elif isinstance(value, list):
            self._table = np.array(value)
        elif isinstance(value, np.ndarray):
            self._table = value
        else:
            NotImplementedError('{}.table.setter() failed with type(table) = {}'.format(self.name, type(value)))
        self.nl_table = np.log(self.table)

    def make(self):
        pass

    def make_table(self):
        pass

    def gradient(self):
        return np.zeros(Params.nparams)

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
        self.nl_table = np.dot(self.params_tab, self.feats.feats)
        self.table = np.exp(-np.clip(self.nl_table, -700, 700))
        # self.table = np.exp(-self.nl_table)

    def gradient(self):
        idx = tuple( v.ivalue for v in self.variables )
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
        kwargs = { feat.name: feat.feats for feat in self.graph.features }
# update with variable values
        vdict = self.graph.vdict(self.variables)
        self.funfeats = np.array([
            self.fun(  _values = np.array(prodict.values()),
                        **dict(kwargs, **prodict)) for prodict in proditer(**vdict)
        ])
        self.funfeats.shape = self.arity + (-1,)

# make/set table
        logger.debug('%s.make_table() - self.funfeats: %s', self.name, str(self.funfeats))
        self.nl_table = np.dot(self.funfeats, self.params.params)
        logger.debug('%s.make_table() - nl_table: %s', self.name, str(self.nl_table))
        self.table = np.exp(-np.clip(self.nl_table, -700, 700))
        # self.table = np.exp(-self.nl_table)
        
    def gradient(self):
        idx = tuple( v.ivalue for v in self.variables)
        pr = self.graph.pr(self.vertex)

        g = np.zeros(Params.nparams)
        g[self.params.pslice] = self.funfeats[idx] - np.tensordot(pr, self.funfeats, self.nvariables)
        return g

