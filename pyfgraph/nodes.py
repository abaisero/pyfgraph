import numpy as np
import numpy.random as rnd

from collections import namedtuple
from itertools import chain

from parametric import Params
from utils.decorators import kwargsdec
from utils.iterators import proditer

from math import log
from operator import mul

import logging, sys
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

        # logger.info('Variable %s', self.name)
        # logger.info(' - domain: %s', self.domain)
        # logger.info(' - idomain: %s', self.idomain)
        # logger.info(' - v2iv: %s', self.v2iv)
        # logger.info(' - iv2v: %s', self.iv2v)

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
        self.arity = tuple( v.arity for v in self.variables )
        self.feats = feats
        self.params = params

        self._log_table = None

    @property
    def log_table(self):
        return self._log_table

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

    def log_value(self, data):
        if self.log_table is None:
            raise Exception
        idx = tuple( v.v2iv[getattr(data.values, v.name)] for v in self.variables )
        return self.log_table[idx]

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
        logger.error('Factor.gradient() is an interface and is not implemented yet.')
        raise NotImplementedError

class TabFactor(Factor):
    def __init__(self, graph, vertex, name, variables):
        super(TabFactor, self).__init__(graph, vertex, name, variables)

    @property
    def table(self):
        return np.exp(self._log_table)

    @table.setter
    def table(self, value):
# TODO check that the values are non-negative
        if isinstance(value, (int, long, float)):
            if self._log_table is None:
                self._log_table = np.empty(self.arity)
            self._log_table.fill(log(value))
        elif isinstance(value, list):
            self._log_table = np.log(value)
        elif isinstance(value, np.ndarray):
            self._log_table = np.log(value)
        else:
            NotImplementedError('{}.table.setter() failed with type(value) = {}'.format(self.name, type(value)))

    @Factor.log_table.setter
    def log_table(self, value):
        if isinstance(value, (int, long, float)):
            if self._log_table is None:
                self._log_table = np.empty(self.arity)
            self._log_table.fill(value)
        elif isinstance(value, list):
            self._log_table = np.array(value)
        elif isinstance(value, np.ndarray):
            self._log_table = value
        else:
            NotImplementedError('{}.log_table.setter() failed with type(value) = {}'.format(self.name, type(value)))

    def make(self):
        pass

    def make_table(self, feats):
        pass

    def gradient(self, data):
        return np.zeros(Params.nparams)

class TabPFactor(Factor):
    def __init__(self, graph, vertex, name, variables):
        super(TabPFactor, self).__init__(graph, vertex, name, variables)
        if self.params is None:
            self.params = self.graph.add(Params, '_{}.param'.format(name), nparams=np.prod(self.arity))

    def make(self):
        self.params_tab = self.params.params.view()
        self.params_tab.shape = self.arity + (-1,)

    def make_table(self):
        self.log_table = self.params_tab

    def gradient(self):
        idx = tuple( v.ivalue for v in self.variables )
        idx_r = np.ravel_multi_index(idx, self.arity)
        g = np.zeros(Params.nparams)
        g[self.params.pslice][idx_r] = 1
        return g

class SimpleFeatFactor(Factor):
    def __init__(self, graph, vertex, name, variables, feats, params = None):
        super(SimpleFeatFactor, self).__init__(graph, vertex, name, variables, feats, params)
        if self.params is None:
            self.params = self.graph.add(Params, '{}.param'.format(name), nparams=self.feats.nfeats)

    def make(self):
        pass

    def make_table(self, data):
        feats = getattr(data.feats, self.feats.name)
        self.log_table = np.dot(feats, self.params.params)

    def log_value(self, data):
        feats = getattr(data.feats, self.feats.name)
        idx = tuple( v.v2iv[getattr(data.values, v.name)] for v in self.variables )
        return np.dot(feats[idx], self.params.params)

    def gradient(self, data):
        feats = getattr(data.feats, self.feats.name)
        idx = tuple( v.v2iv[getattr(data.values, v.name)] for v in self.variables )

        # logger.debug('%s.gradient():', self.name)
        # logger.debug(' * totparams: %s', Params.nparams)
        # logger.debug(' * pslice: %s', self.params.pslice)
        # logger.debug(' * feats.shape: %s', feats.shape)
        # logger.debug(' * idx: %s', idx)

        g = np.zeros(Params.nparams)
        g[self.params.pslice] = feats[idx]
        return g

        # pr = self.graph.pr(self.vertex)

        # ttable = np.zeros(self.arity)
        # ttable[idx] = 1

        # # logger.debug('%s.gradient():', self.name)
        # # logger.debug(' * ttable: %s', ttable)
        # # logger.debug(' * pr:     %s', pr)
        # # logger.debug(' * first:  %s', np.kron(ttable, self.feats.feats))
        # # logger.debug(' * second: %s', np.kron(pr, self.feats.feats))

        # g = np.zeros(Params.nparams)
        # # g[self.params.pslice] = np.kron(ttable-pr, self.feats.feats).ravel()
        # g[self.params.pslice] = np.kron(pr-ttable, self.feats.feats).ravel()
        # return g


class FeatFactor(Factor):
    def __init__(self, graph, vertex, name, variables, feats, params = None):
        super(FeatFactor, self).__init__(graph, vertex, name, variables, feats, params)
        if self.params is None:
            self.params = self.graph.add(Params, '{}.param'.format(name), nparams=np.prod(self.arity)*self.feats.nfeats)
            # self.params = self.graph.add(Params, '{}.param'.format(name), nparams=self.feats.nfeats)

    def make(self):
        self.params_tab = self.params.params.view()
        self.params_tab.shape = self.arity + (-1,)
        pass

    def make_table(self, feats):
        feats = getattr(feats, self.feats.name)
        self._log_table = np.einsum('...i,...i', feats, self.params_tab)
        # self.log_table = np.dot(feats, self.params.params)

    def log_value(self, data):
        # feats = getattr(data.feats, self.feats.name)
        idx = tuple( v.v2iv[getattr(data.values, v.name)] for v in self.variables )
        # return np.dot(feats[idx], self.params.params)
        return self.log_table[idx]

    def Efeats(self, data):
        feats = getattr(data.feats, self.feats.name)
        pr = self.graph.pr(self.vertex)

        g = np.zeros(Params.nparams)
        g[self.params.pslice] = np.einsum('...,...i', pr, feats).ravel()
        return g

    def gradient(self, data):
        feats = getattr(data.feats, self.feats.name)
        idx = tuple( v.v2iv[getattr(data.values, v.name)] for v in self.variables )
        # print feats
        # print idx
        # print self.params.pslice
        # idx_r = np.ravel_multi_index(idx, self.arity)
        # print idx_r

        # # logger.debug('%s.gradient():', self.name)
        # # logger.debug(' * totparams: %s', Params.nparams)
        # # logger.debug(' * pslice: %s', self.params.pslice)
        # # logger.debug(' * feats.shape: %s', feats.shape)
        # # logger.debug(' * idx: %s', idx)

        # g = np.zeros(Params.nparams)
        # g[self.params.pslice] = feats[idx]
        # return g

        pr = self.graph.pr(self.vertex)
        ttable = np.zeros(self.arity)
        ttable[idx] = 1

        g = np.zeros(Params.nparams)
        # g[self.params.pslice] = np.kron(pr-ttable, feats[idx]).ravel()
        g[self.params.pslice] = np.einsum('...,...i', pr-ttable, feats).ravel()
        return g

class FunFactor(Factor):
    def __init__(self, graph, vertex, name, variables, feats = None, params = None, nfunfeats = None):
        super(FunFactor, self).__init__(graph, vertex, name, variables, feats, params)

        if self.feats is None and nfunfeats is None:
            raise Exception('FunFactor.__init__() requires either `feats` or `nfunfeats` to be specified')
        if self.feats is not None and nfunfeats is not None and self.feats.nfeats != nfunfeats:
            raise Exception('FunFactor.__init__() receives incompatible `feats` and `nfunfeats` parameters')

        self.params = self.graph.add(Params, '_{}.param'.format(name), nparams=nfunfeats or self.feats.nfeats)
        self._fun = None
        self._funfeats = None

    @property
    def fun(self):
        return self._fun

    @fun.setter
    def fun(self, value):
        if callable(value):
            self._fun = value
        else:
            raise Exception('{}.fun.setter() requires a callable as argument.'.format(self.name))

    def make(self):
        pass

    def make_table(self):
        if self._funfeats is None:
            self._funfeats = {}
        self.graph.feats.flags.writeable = False
        h = hash(self.graph.feats.data)
        self.graph.feats.flags.writeable = True

        if h not in self._funfeats:
# make/set features
            kwargs = { feat.name: feat.feats for feat in self.graph.features }
# update with variable values
            vdict = self.graph.vdict(self.variables)
            all_names = chain(
                [ feat.name for feat in self.graph.features ],  # feature names
                vdict.keys(),                                   # variable names
                ['values'],                                     # 'values' string
            )

            State = namedtuple('State', ', '.join(all_names))
            ff = np.array([
                self.fun(State(
                    values = np.array(prodict.values()),
                    **dict(kwargs, **prodict)
                )) for prodict in proditer(**vdict)
            ])
            ff.shape = self.arity + (-1,)
            self._funfeats[h] = ff

        self.funfeats = self._funfeats[h]
        
# make/set table
        # logger.debug('%s.make_table() - self.funfeats: %s', self.name, str(self.funfeats))
        self.log_table = np.dot(self.funfeats, self.params.params)
        # self.table = np.exp(self.log_table)
        
    def gradient(self):
        idx = tuple( v.ivalue for v in self.variables)
        pr = self.graph.pr(self.vertex)

        g = np.zeros(Params.nparams)
        # g[self.params.pslice] = self.funfeats[idx] - np.tensordot(pr, self.funfeats, self.nvariables)
        g[self.params.pslice] = np.tensordot(pr, self.funfeats, self.nvariables) - self.funfeats[idx] 
        return g
