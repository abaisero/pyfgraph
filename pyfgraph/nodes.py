import numpy as np
import numpy.random as rnd

import logging, log
logger = logging.getLogger()

from params import Params

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
        self.value = None

class Factor(Node):
    def __init__(self, vertex, name, variables, params=None):
        super(Factor, self).__init__(vertex, name)
        self.variables = variables if isinstance(variables, tuple) else (variables,)
        self.arity = tuple( v.arity for v in self.variables )
        self.table = np.empty(self.arity)
        self.params = params

    def setTable(self, table):
        if isinstance(table, (int ,long, float)):
            self.table.fill(table)
        elif isinstance(table, np.ndarray):
            self.table = table
        else:
            NotImplementedError('setTable failed with type(table) = {}'.format(type(table)))

    def value(self, idx = None):
        if idx is None:
            idx = tuple( v.value for v in self.variables )
        return self.table[idx]

    def gradient(self):
        raise NotImplementedError

class FFactor(Factor):
    def __init__(self, *args, **kwargs):
        super(FFactor, self).__init__(*args, **kwargs)
        self.nfeats = 0
        self.feats = None
        # self.params = None
        # self.nparams = 0
        self.pshape = self.arity + (-1,)

        # self.params = kwargs['params']

    # def gradient(self, X):
    #     print 'computing gradient'

    def setFeats(self, feats):
        # print 'feats: {}'.format(feats)
        self.feats = feats
        self.nfeats = len(feats)

    # def getParams(self):
    #     return self.params.ravel()

    # def setParams(self, params, nfeats = None):
    #     if isinstance(params, np.ndarray):
    #         self.params = params.reshape(self.pshape)
    #     elif params == 'random':
    #         self.params = rnd.randn(*tuple(self.arity + (self.nfeats,)))
    #     else:
    #         raise NotImplementedError('setParams with {} not done yet').format(params)
    #     self.nparams = self.params.size

    def getParams(self):
        return self.params

    def setParams(self, params, nfeats = None):
        if isinstance(params, np.ndarray):
            print 'setParams: {}'.format(params)
            self.params[:] = params.ravel()
        elif params == 'random':
            self.params[:] = .1 * rnd.randn(self.params.size)
        else:
            raise NotImplementedError('setParams with {} not done yet').format(params)
        self.nparams = self.params.size

    def makeTable(self):
# TODO this is the value which the table will take for specific values
        # idx = tuple( v.value for v in self.variables )
        # return np.dot(self.feats, self.params[idx])

        if self.params is None:
            self.setParams('random')
        self.nl_table = np.dot(self.params_tab, self.feats)
        self.table = np.exp(-self.nl_table)

        # for items in itertools.product(*[ range(v.arity) for v in self.variables ]):
        #     print items
        # self.table = 
        # self.phi = np.array([ v.value ])

    def gradient(self):
        idx = tuple( v.value for v in self.variables )
        pr = self.graph.pr(self.vertex, with_l1_norm=False)
        # print '==='
        # print 'idx: ', idx
        # print 'feats: ', self.feats
        # print 'pr: ', pr
        # return self.feats[idx] - np.tensordot(pr, self.feats)
        ttable = np.zeros(self.arity)
        ttable[idx] = 1
        # print 'ttable:'
        # print ttable

        logger.debug('%s.gradient():', self.name)
        logger.debug(' * ttable: %s', ttable)
        logger.debug(' * pr:     %s', pr)
        logger.debug(' * first:  %s', np.kron(ttable, self.feats))
        logger.debug(' * second: %s', np.kron(pr, self.feats))

        g = np.zeros(Params.tot_nparams)
        g[self.pslice] = (np.kron(ttable, self.feats) - np.kron(pr, self.feats)).ravel()
        return g
        # return (np.kron(ttable, self.feats) - np.kron(pr, self.feats)).ravel()

