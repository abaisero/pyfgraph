import inspect
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

from pyfgraph.parametric import Feats, Params
from pyfgraph.nodes import Node, Variable, Factor, FeatFactor, FunFactor
import pyfgraph.utils.log as log
import pyfgraph.algo as algo

import logging
logger = logging.getLogger(__name__)

class FactorGraph(object):
    def __init__(self):
        self.graph = Graph(directed=False)
        self.variables = []
        self.factors = []
        self.features = []
        self.parameters = []
        self.logZ = None

        self._feats = None
        self._params = None

# properties
        self.vp_type = self.graph.new_vertex_property("string")
        self.vp_name = self.graph.new_vertex_property("string")
        self.vp_shape = self.graph.new_vertex_property("string")
        self.vp_color = self.graph.new_vertex_property("string")
        self.vp_size = self.graph.new_vertex_property("int")
        self.vp_arity = self.graph.new_vertex_property("int")
        self.vp_table = self.graph.new_vertex_property("object")
        self.vp_table_inputs = self.graph.new_vertex_property("object")

        self.vp_node = self.graph.new_vertex_property("object")

        self.ep_sp_msg_fv = self.graph.new_edge_property("object")
        self.ep_sp_msg_fv_lnc = self.graph.new_edge_property("double")
        self.ep_sp_msg_vf = self.graph.new_edge_property("object")
        self.ep_sp_msg_vf_lnc = self.graph.new_edge_property("double")

        self.ep_mp_msg_fv = self.graph.new_edge_property("object")
        self.ep_mp_msg_fv_lnc = self.graph.new_edge_property("double")
        self.ep_mp_msg_vf = self.graph.new_edge_property("object")
        self.ep_mp_msg_vf_lnc = self.graph.new_edge_property("double")

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
        elif isinstance(value, np.ndarray) or isinstance(value, list):
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
        self.feats = np.empty(Feats.nfeats)
        for f in self.features:
            f.make(self.feats)
            logger.info('Feat %s: %s (nfeats: %s)', f.name, f.fslice, f.nfeats)

        self.params = np.empty(Params.nparams)
        for p in self.parameters:
            p.make(self.params)
            logger.info('Param %s: %s (nparams: %s)', p.name, p.pslice, p.nparams)

        for f in self.factors:
            self.vp_node[f].make()

    # def make(self, feats=False, params=False, graph=False):
    #     if feats:
    #         self.feats = np.empty(Feats.nfeats)
    #         for f in self.features:
    #             f.make(self.feats)
    #             logger.info('Feat %s: %s (nfeats: %s)', f.name, f.fslice, f.nfeats)
    #     if params:
    #         self.params = np.empty(Params.nparams)
    #         for p in self.parameters:
    #             p.make(self.params)
    #             logger.info('Param %s: %s (nparams: %s)', p.name, p.pslice, p.nparams)
    #     if graph:
    #         # self.feats = np.empty(Feats.nfeats)
    #         # self.params = np.empty(Params.nparams)

# # probably not necessary anymore
    #         for f in self.factors:
    #             self.vp_node[f].make()
    
    def plot(self):
        graph_draw(
            self.graph,
            vertex_text=self.vp_name,
            vertex_shape=self.vp_shape,
            vertex_color="black",
            vertex_fill_color=self.vp_color,
            vertex_size=self.vp_size
        )

    def check_message_passing(self):
        print 'Checking the message passing. For each variable, all the rows should be the same.'
        for v in self.variables:
            print 'Variable', self.vp_name[v]
            print np.sum([ self.ep_sp_msg_fv_lnc[e] + np.log(self.ep_sp_msg_fv[e]) for e in v.out_edges() ], axis = 0)
            for e in v.out_edges():
                msg_vf = self.ep_sp_msg_vf_lnc[e] + np.log(self.ep_sp_msg_vf[e])
                msg_fv = self.ep_sp_msg_fv_lnc[e] + np.log(self.ep_sp_msg_fv[e])
                print msg_vf + msg_fv

    def write(self):
        print 'Vertices:'
        for v in self.graph.vertices():
            print 'vertex {}'.format(v)
            print ' * type: {}'.format(self.vp_type[v])
            print ' * in_degree: {}'.format(v.in_degree())
            print ' * out_degree: {}'.format(v.out_degree())
            print ' * table.shape: {}'.format(None if self.vp_table[v] is None else self.vp_table[v].shape)
            print ' * table_inputs: {}'.format(self.vp_table_inputs[v])
            print ' * arity: {}'.format(self.vp_arity[v])

    def pr(self, vertex, with_log_norm=False):
# # TODO OH NO!!!! recursion!!!
# TODO write a "message passing" function, which runs the algorithm, detects when it is time to run it, etc....
# TODO factorize!!
#         algo.message_passing(self, 'sum-product')

        s_vtype = self.vp_type[vertex]
        if s_vtype == 'variable':
            e = vertex.out_edges().next()
            pr = self.ep_sp_msg_vf[e] * self.ep_sp_msg_fv[e]
            prs = pr.sum()
            lprs = math.log(prs)

            if with_log_norm:
                lnc = self.ep_sp_msg_vf_lnc[e] + self.ep_sp_msg_fv_lnc[e] + lprs
        elif s_vtype == 'factor':
            msgs = { self.vp_name[e.target()]: self.ep_sp_msg_vf[e] for e in vertex.out_edges() }
            msgs = [ msgs[n] for n in self.vp_table_inputs[vertex] ]
            pr = self.vp_table[vertex] * reduce(np.multiply, np.ix_(*msgs))
            prs = pr.sum()

            if with_log_norm:
                lnc = np.array([ self.ep_sp_msg_vf_lnc[e] for e in vertex.out_edges() ]).sum() + lprs
        else:
            raise Exception('variable type error: {}'.format(s_vtype))

        pr /= prs
        return (pr, lnc) if with_log_norm else pr

    def max(self):
        # log.log_messages(self, ('max-product',))

        v = self.variables[0]
        e = v.out_edges().next()
        vf_lnc = self.ep_mp_msg_vf_lnc[e]
        fv_lnc = self.ep_mp_msg_fv_lnc[e]
        return math.exp(vf_lnc + fv_lnc) * ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).max()

    def argmax(self):
        # log.log_messages(self, ('max-product',))
        # ivalues = [None] * self.nvariables
        values = [None] * self.nvariables

        for i, v in enumerate(self.variables):
            e = v.out_edges().next()
            ivalue = (self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e]).argmax()
            # ivalues[i] = ivalue
            values[i] = self.vp_node[v].iv2v[ivalue]

        return values

    @property
    def feats(self):
        return self._feats
    
    @feats.setter
    def feats(self, value):
        if isinstance(value, np.ndarray):
            if value.size != Feats.nfeats:
                raise Exception('The specified FactorGraph requires {} feats, but you provided {}'.format(Feats.nfeats, value.size))
            if value.size == self.nfeats:
                self._feats[:] = value.ravel()
            else:
                self._feats = value.ravel().copy()
        else:
            raise NotImplementedError('feats.setter with object type {} not defined.').format(type(value))

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
    
    def viterbi(self, data = None, params = None):
        if params is not None:
            self.params = params

        vit = []
        for x in data['X']:
            self.values = None
            self.feats = x
            for f in self.factors:
                self.vp_node[f].make_table()
            algo.message_passing(self, 'max-product')
            vit.append(self.argmax())

        return vit

    def nll(self, data = None, params = None):
        if data is None:
# Assume data is already set, 
            # return -np.log([ self.vp_node[f].l() for f in self.factors ]).sum()
            logZ = self.logZ
            nll = np.sum( self.vp_node[f].nl_value() for f in self.factors )
            logger.debug('logZ: %s', self.logZ)
            logger.debug('nll: %s', nll)
            return self.logZ + nll

        if params is not None:
            self.params = params

# TODO implement later today
        Feats.extend(data)

        X, Y = data['X'], data['Y']

        ndata = X.shape[0]

        nll = np.empty(ndata)
        for i, x, y in itt.izip(itt.count(), X, Y):
            self.feats = x
            self.values = None
            self.values = y

            for f in self.factors:
                self.vp_node[f].make_table()
            # log.log_params(self)
            # log.log_tables(self)
            algo.message_passing(self, 'sum-product')

            nll[i] = self.nll()
        return nll

    def dnll(self, data = None, params = None):
        if data is None:
# Assume data is already set, 
            return np.array([ self.vp_node[f].gradient() for f in self.factors ]).sum(axis=0)

        if params is not None:
            self.params = params

# TODO implement later today
        Feats.extend(data)

        X, Y = data['X'], data['Y']
        
        ndata = X.shape[0]

        dnll = np.empty((ndata, self.nparams))
        for i, x, y in itt.izip(itt.count(), X, Y):
            self.feats = x
            self.values = None
            self.values = y
            for f in self.factors:
                self.vp_node[f].make_table()
            # log.log_params(self)
            # log.log_tables(self)
            algo.message_passing(self, 'sum-product')

            dnll[i] = self.dnll()
        return dnll

    def train(self, data):
        # if 'Y_kwlist' not in data:
        #     data['Y_kwlist'] = [ { self.vp_node[var].name: value for var, value in zip(self.variables, y) } for y in data['Y'] ]

        self.params = 'random'

        def fun(params, data):
            cost = self.nll(data = data, params = params).sum() 
            reg = 1. * la.norm(params, ord=1)
            logger.debug('Objective Function: %s ( %s + %s )', cost+reg, cost, reg)
            return cost + reg

        def jac(params, data):
            dcost = self.dnll(data = data, params = params).sum(axis=0)
            dreg = 1. * np.sign(params)
            logger.debug('Objective Jacobian: %s ( %s + %s )', dcost+dreg, dcost, dreg)
            return dcost + dreg

        i = [0]
        def callback(params):
            logger.info('opt.minimize.params: %s', str(params))
            # if i[0] == 100:
            #     sys.exit(-1)
            # i[0] += 1

        logger.info('BEGIN Optimization')
        res = opt.minimize(
            fun  = fun,
            x0   = self.params,
            # method='nelder-mead',
            jac  = jac,
            args = (data,),
            # callback = callback
        )
        logger.info('END Optimization')
        logger.info('Training result: %s', res)
        # if not res.success:
        #     raise Exception('Failed to train!')
        self.params = res.x

