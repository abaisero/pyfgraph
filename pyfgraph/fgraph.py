import nphelper

import operator
import inspect
from collections import namedtuple
import sys

import sys, warnings
warnings.filterwarnings('error')

from graph_tool.all import *
from graph_tool.util import find_vertex

import scipy.optimize as opt
import itertools as itt

import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from scipy.misc import logsumexp

from pyfgraph.parametric import Feats, Params
from pyfgraph.nodes import Node, Variable, Factor, FeatFactor, FunFactor
import pyfgraph.utils.log as log
import pyfgraph.algo as algo

import logging
logger = logging.getLogger(__name__)

class FactorGraph(object):
    def __init__(self):
        self.graph = Graph(directed=False)
        self.clean()

# properties
        self.vp_type = self.graph.new_vertex_property("string")
        self.vp_name = self.graph.new_vertex_property("string")
        self.vp_shape = self.graph.new_vertex_property("string")
        self.vp_color = self.graph.new_vertex_property("string")
        self.vp_size = self.graph.new_vertex_property("int")
        self.vp_arity = self.graph.new_vertex_property("int")
        self.vp_log_table = self.graph.new_vertex_property("object")
        self.vp_table_inputs = self.graph.new_vertex_property("object")

        self.vp_node = self.graph.new_vertex_property("object")

        self.ep_sp_log_msg_fv = self.graph.new_edge_property("object")
        self.ep_sp_log_msg_vf = self.graph.new_edge_property("object")

        self.ep_mp_log_msg_fv = self.graph.new_edge_property("object")
        self.ep_mp_log_msg_vf = self.graph.new_edge_property("object")

    def clean(self):
        # self.feats = None

        self._params = None
        self._made = False
        self._trained = False

        self.variables = []
        self.factors = []
        self.features = []
        self.parameters = []
        self.logZ = None

        self.check_gradient = False

        Feats.clean()
        Params.clean()

    @property
    def made(self):
        return self._made

    @property
    def trained(self):
        return self._trained

    @property
    def nvariables(self):
        return len(self.variables)

    @property
    def values(self):
        return [ self.vp_node[v].value for v in self.variables ]

    @values.setter
    def values(self, value):
        if value is None:
            for v in self.variables:
                self.vp_node[v].value = None
        elif isinstance(value, self.Values):
            for v in self.variables:
                self.vp_node[v].value = getattr(value, self.vp_node[v].name)
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            raise Exception('This should not happen anymore.')
            for var, val in zip(self.variables, value):
                self.vp_node[var].value = val
        elif isinstance(value, dict):
            raise Exception('This should not happen anymore.')
            for k, v in value.iteritems():
                vertices = find_vertex(self.graph, self.vp_name, k)
                if len(vertices) is 1:
                    self.vp_node[vertices[0]].value = v
                elif len(vertices) > 1:
                    raise Exception('This should not happen.')
            # log.log_values(self)
        else:
            raise NotImplementedError('values.setter with object type {} not defined.'.format(type(value)))

    def vdict(self, variables=None):
        """ returns a dictionary with 'variable' -> 'domain' mappings """
        if variables is None:
            return { self.vp_node[v].name: self.vp_node[v].domain for v in self.variables }
        return { v.name: v.domain for v in variables }

    def add(self, cls, *args, **kwargs):
        if issubclass(cls, Node):
            vertex = self.graph.add_vertex()

            c = cls(self, vertex, *args, **kwargs)
            self.vp_node[vertex] = c
            self.vp_name[vertex] = c.name

            if issubclass(cls, Variable):
                self.vp_type[vertex] = 'variable'
                self.vp_shape[vertex] = 'circle'
                self.vp_color[vertex] = 'white'
                self.vp_size[vertex] = 50

# variable-specific properties
                self.vp_arity[vertex] = c.arity
                self.variables.append(vertex)
            elif issubclass(cls, Factor):
                self.vp_type[vertex] = 'factor'
                self.vp_shape[vertex] = 'square'
                self.vp_color[vertex] = 'black'
                self.vp_size[vertex] = 30

# factor-specific properties
                self.vp_table_inputs[vertex] = [ self.vp_name[variable.vertex] for variable in c.variables ]
                self.factors.append(vertex)

# adding all edges of this factor
                for variable in c.variables:
                    self.graph.add_edge(vertex, variable.vertex)
        elif issubclass(cls, Feats):
            c = cls(*args, **kwargs)
            self.features.append(c)
        elif issubclass(cls, Params):
            c = cls(*args, **kwargs)
            self.parameters.append(c)
        else:
            raise Exception('what are you doing..')

        return c

    def make(self):
        # self.feats = np.empty(Feats.nfeats)
        # for f in self.features:
        #     f.make(self.feats)
        #     logger.info('Feat %s: %s (nfeats: %s)', f.name, f.fslice, f.nfeats)

        self.params = np.empty(Params.nparams)
        for p in self.parameters:
            p.make(self.params)
            logger.info('Param %s: %s (nparams: %s)', p.name, p.pslice, p.nparams)

        for f in self.factors:
            self.vp_node[f].make()

# Data containers
        self.Feats  = namedtuple('Feats', [ f.name for f in self.features ])
        self.Values = namedtuple('Values', [ self.vp_node[v].name for v in self.variables ])
        self.Values.__new__.__defaults__ = (None,) * self.nvariables

        self.Data = namedtuple('Data', 'feats, values')

        self._made = True

    def plot(self):
        graph_draw(
            self.graph,
            vertex_text=self.vp_name,
            vertex_shape=self.vp_shape,
            vertex_color="black",
            vertex_fill_color=self.vp_color,
            vertex_size=self.vp_size
        )

    def log_pr(self, vertex):
        """ log likelihood, without the normalization constant """
        s_vtype = self.vp_type[vertex]
        if s_vtype == 'variable':
            e = vertex.out_edges().next()
            log_pr = self.ep_sp_log_msg_vf[e] + self.ep_sp_log_msg_fv[e]
        elif s_vtype == 'factor':
            msgs = { self.vp_name[e.target()]: self.ep_sp_log_msg_vf[e] for e in vertex.out_edges() }
            msgs = [ msgs[n] for n in self.vp_table_inputs[vertex] ]
            log_pr = self.vp_log_table[vertex] + reduce(np.add, np.ix_(*msgs))
        else:
            raise Exception('variable type error: {}'.format(s_vtype))

        return log_pr

    def pr(self, vertex):
        """ likelihood, with the normalization constant """
        log_pr = self.log_pr(vertex)
        return np.exp(log_pr - self.logZ)

    def max(self):
        # log.log_messages(self, ('max-product',))
        v = self.variables[0]
        e = v.out_edges().next()
        return math.exp(( self.ep_mp_log_msg_vf[e] + self.ep_mp_log_msg_fv[e] ).max())

    def argmax(self):
        # log.log_messages(self, ('max-product',))
        values = {}
        for v in self.variables:
            vnode = self.vp_node[v]
            e = v.out_edges().next()
            ivalue = (self.ep_mp_log_msg_vf[e] + self.ep_mp_log_msg_fv[e]).argmax()
            values[vnode.name] = vnode.iv2v[ivalue]

        return self.Values(**values)

        # # log.log_messages(self, ('max-product',))
        # values = [None] * self.nvariables
        # for i, v in enumerate(self.variables):
        #     e = v.out_edges().next()
        #     ivalue = (self.ep_mp_log_msg_vf[e] + self.ep_mp_log_msg_fv[e]).argmax()
        #     values[i] = self.vp_node[v].iv2v[ivalue]
        # return values

    @property
    def nfeats(self):
        if isinstance(self._feats, np.ndarray):
            return self._feats.size
        return None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        if isinstance(value, np.ndarray):
            if self.nparams == value.size:
                self._params[:] = value.ravel()
            else:
                self._params = value.ravel()
        elif value == 'random':
            self._params[:] = .01 * rnd.randn(self._params.size)
        else:
            raise NotImplementedError('params.setter with object type {} not defined.').format(type(value))

    @property
    def nparams(self):
        if isinstance(self._params, np.ndarray):
            return self._params.size
        return None

    def other_pr(self, v, data):
        _data = data if isinstance(data, list) else [data]
        if any((not isinstance(d, self.Data) for d in _data)):
            raise Exception('Data type not understood')

        ndata = len(_data)
        pr = [None]*ndata
        for i, d in enumerate(_data):
            self.values = d.values
            # self.feats = d.feats

            for f in self.factors:
                self.vp_node[f].make_table(d.feats)
            algo.message_passing(self, 'sum-product')
            pr[i] = self.pr(v.vertex)

        return pr if isinstance(data, list) else pr[0]
    
    def viterbi(self, data = None, params = None):
        _data = data if isinstance(data, list) else [data]
        if any((not isinstance(d, self.Data) for d in _data)):
            raise Exception('Data type not understood')

        if params is not None:
            self.params = params

# TODO copy input and add new values
        ndata = len(_data)
        vit = [None]*ndata
        for i, d in enumerate(_data):
            self.values = d.values
            # self.feats = d.feats

            for f in self.factors:
                self.vp_node[f].make_table(d.feats)
            algo.message_passing(self, 'max-product')
            vit[i] = d._replace(values=self.argmax())

        return vit if isinstance(data, list) else vit[0]

    def nll(self, data = None, params = None):
        if any((not isinstance(d, self.Data) for d in data)):
            raise Exception('Data type not understood')
        if any((None in d.values for d in data)):
            raise Exception('Data.values should be fully determined.')

        if params is not None:
            self.params = params

        ndata = len(data)
        nll = np.empty(ndata)
        for i, d in enumerate(data):
            # self.feats = d.feats
            # self.values = d.values

            for f in self.factors:
                self.vp_node[f].make_table(d.feats)
            # log.log_params(self)
            # log.log_tables(self)
            algo.message_passing(self, 'sum-product')

            nll[i] = self.logZ - np.sum( self.vp_node[f].log_value(d) for f in self.factors )
        return nll

    def dnll(self, data = None, params = None):
        if any((not isinstance(d, self.Data) for d in data)):
            raise Exception('Data type not understood')
        if any((None in d.values for d in data)):
            raise Exception('Data.values should be fully determined.')

        if params is not None:
            self.params = params

        ndata = len(data)
        dnll = np.empty((ndata, self.nparams))
        for i, d in enumerate(data):
            # self.feats = d.feats
            # self.values = d.values

            for f in self.factors:
                self.vp_node[f].make_table(d.feats)
            # log.log_params(self)
            # log.log_tables(self)
            algo.message_passing(self, 'sum-product')

            dnll[i] = np.sum([
                self.vp_node[f].gradient(d) for f in self.factors
            ], axis = 0)
        return dnll
    
    def ddnll(self, data = None, params = None):
        if any((not isinstance(d, self.Data) for d in data)):
            raise Exception('Data type not understood')
        if any((None in d.values for d in data)):
            raise Exception('Data.values should be fully determined.')

        if params is not None:
            self.params = params

        ndata = len(data)
        ddnll = np.empty((ndata, self.nparams, self.nparams))
        for i, d in enumerate(data):
            # self.feats = d.feats
            # self.values = d.values

            for f in self.factors:
                self.vp_node[f].make_table(d.feats)
            # log.log_params(self)
            # log.log_tables(self)
            algo.message_passing(self, 'sum-product')

            Efeats = [ list(self.vp_node[f].Efeats(d)) for f in self.factors ]
            ddnll[i] = np.sum([
                np.outer(ef1, ef2) for ef2 in Efeats for ef1 in Efeats
            ], axis=0)

        return ddnll
    

    def train(self, data):
        self.params = 'random'

        l = 1.
        def fun(params, data):
            cost = self.nll(data = data, params = params).sum() 
            reg = l * la.norm(params, ord=1)
            f = cost+reg

            logger.debug('Objective Function: %s ( %s + %s )', f, cost, reg)
            return f

        def jac(params, data):
            dcost = self.dnll(data = data, params = params).sum(axis=0)
            dreg = l * np.sign(params)
            j = dcost+dreg

            logger.debug('Objective Jacobian: %s ( %s + %s )', j, dcost, dreg)
            return j
        
        def hess(params, data):
            h = self.ddnll(data = data, params = params).sum(axis=0)

            logger.debug('Objective Hessian: %s', h)
            return h

        err_grad = []
        def callback(params):
            logger.info('opt.minimize.params: %s', str(params))
            err = opt.check_grad(fun, jac, params, data)
            logger.debug('err_grad: %s', err)
            err_grad.append(err)

        logger.info('BEGIN Optimization')
        # xopt = opt.fmin_bfgs(
        #     f = fun,
        #     x0   = self.params,
        #     fprime  = jac,
        #     args = (data,),
        #     callback = callback if self.check_gradient else None,
        #     gtol = 2,
        #     norm = np.inf
        # )
        res = opt.minimize(
            fun  = fun,
            x0   = self.params,
            jac  = jac,
            # hess = hess,
            # method = 'Newton-CG',
            args = (data,),
            callback = callback if self.check_gradient else None,
            options = {'gtol': 5, 'norm': np.inf},
        )
        logger.info('END Optimization')
        logger.info('Training result: %s', res)
        # if not res.success:
        #     raise Exception('Failed to train!')
        self.params = res.x

        self._trained = True

        return np.array(err_grad) if err_grad else None

