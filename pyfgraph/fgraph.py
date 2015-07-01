import inspect

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

from parametric import Feats, Params
from nodes import Node, Variable, Factor, FeatFactor, FunFactor
import algo

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
    def values(self):
        return [ self.vp_node[v].value for v in self.variables ]

    @values.setter
    def values(self, value):
        if value is None:
            for v in self.variables:
                self.vp_node[v].value = None
        elif isinstance(value, dict):
            for k, v in value.iteritems():
                vertices = find_vertex(self.graph, self.vp_name, k)
                if len(vertices) is 1:
                    self.vp_node[vertices[0]].value = v
                elif len(vertices) > 1:
                    raise Exception('This should not happen.')
        else:
            raise NotImplementedError('values.setter with object type {} not defined.').format(type(value))

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

    # def clear_msgs(self):
    #     for e in self.graph.edges():
    #         self.ep_sp_msg_fv[e] = None
    #         self.ep_sp_msg_vf[e] = None

    #         self.ep_sp_msg_fv_nc[e] = 0.
    #         self.ep_sp_msg_vf_nc[e] = 0.

    #         self.ep_mp_msg_fv[e] = None
    #         self.ep_mp_msg_vf[e] = None

    #         self.ep_mp_msg_fv_nc[e] = 0.
    #         self.ep_mp_msg_vf_nc[e] = 0.
            
    #         self.Z = None
    #         self.logZ = None

    # def pass_msgs(self, s, t):
    #     snode = self.vp_node[s]
    #     tnode = self.vp_node[t]
    #     logger.debug('passing %s -> %s', snode, tnode)

    #     s_vtype = self.vp_type[s]
    #     if s_vtype == 'variable':
    #         arity = self.vp_arity[s]
    #         sp_msg, sp_msg_nc = np.ones(arity), 1
    #         mp_msg, mp_msg_nc = np.ones(arity), 1

    #         logger.debug(' * sp_msg:    %s', sp_msg)
    #         logger.debug(' * sp_msg_nc: %s', sp_msg_nc)
    #         logger.debug(' * mp_msg:    %s', mp_msg)
    #         logger.debug(' * mp_msg_nc: %s', mp_msg_nc)
    #         neighbours = list(s.out_neighbours())
    #         neighbours.remove(t)
    #         for neigh in neighbours:
    #             e = self.graph.edge(neigh, s)
    #             sp_msg    *= self.ep_sp_msg_fv[e]
    #             sp_msg_nc *= self.ep_sp_msg_fv_nc[e]
    #             mp_msg    *= self.ep_mp_msg_fv[e]
    #             mp_msg_nc *= self.ep_mp_msg_fv_nc[e]
    #             logger.debug('   -------')
    #             logger.debug(' * sp_msg:    %s', sp_msg)
    #             logger.debug(' * sp_msg_nc: %s', sp_msg_nc)
    #             logger.debug(' * mp_msg:    %s', mp_msg)
    #             logger.debug(' * mp_msg_nc: %s', mp_msg_nc)

    #         e = self.graph.edge(s, t)

    #         sp_msg_nc *= sp_msg.sum()
    #         self.ep_sp_msg_vf[e] = sp_msg/sp_msg.sum()
    #         self.ep_sp_msg_vf_nc[e] = sp_msg_nc

    #         mp_msg_nc *= mp_msg.sum()
    #         self.ep_mp_msg_vf[e] = mp_msg/mp_msg.sum()
    #         self.ep_mp_msg_vf_nc[e] = mp_msg_nc
    #     elif s_vtype  == 'factor':
    #         nname = self.vp_name[t]
    #         msgs = { nname: np.ones(self.vp_arity[t]) }
    #         msg_nc = 1
    #         logger.debug(' * in_msg:    %s', msgs[nname])
    #         logger.debug(' * in_msg_nc: %s', msg_nc)

    #         neighbours = list(s.out_neighbours())
    #         # neighbours.remove(t)
    #         for neigh in filter(lambda neigh: neigh != t, neighbours):
    #             nname = self.vp_name[neigh]
    #             e = self.graph.edge(neigh, s)
    #             msgs[nname] = self.ep_mp_msg_vf[e]
    #             msg_nc *= self.ep_mp_msg_vf_nc[e]
    #             logger.debug(' * in_msg:    %s', self.ep_mp_msg_vf[e])
    #             logger.debug(' * in_msg_nc: %s', self.ep_mp_msg_vf_nc[e])

    #         msgs = [ msgs[n] for n in self.vp_table_inputs[s] ]

    #         logger.debug(' * factor: %s', self.vp_table[s])

    #         prod = self.vp_table[s] * reduce(np.multiply, np.ix_(*msgs))
    #         axis = self.vp_table_inputs[s].index(self.vp_name[t])
    #         negaxis = tuple(filter(lambda a: a!= axis, range(len(neighbours))))
    #         sp_msg = prod.sum(axis = negaxis)
    #         mp_msg = prod.max(axis = negaxis)

    #         e = self.graph.edge(s, t)

    #         sp_msg_sum = sp_msg.sum()
    #         self.ep_sp_msg_fv_nc[e] = msg_nc * sp_msg_sum
    #         self.ep_sp_msg_fv[e] = sp_msg/sp_msg_sum

    #         mp_msg_sum = mp_msg.sum()
    #         self.ep_mp_msg_fv_nc[e] = msg_nc * mp_msg_sum
    #         self.ep_mp_msg_fv[e] = mp_msg/mp_msg_sum

    #         logger.debug(' * out_msg_sp:    %s', self.ep_sp_msg_fv[e])
    #         logger.debug(' * out_msg_sp_nc: %s', self.ep_sp_msg_fv_nc[e])

    #         logger.debug(' * out_msg_mp:    %s', self.ep_mp_msg_fv[e])
    #         logger.debug(' * out_msg_mp_nc: %s', self.ep_mp_msg_fv_nc[e])
    #     else:
    #         raise Exception('variable type error: {}'.format(s_vtype))

    def print_marginals(self, v):
        print 'Marginal Probabilities Variables'
        print 'v: ', v
        print 'variable: {}'.format(self.vp_name[v])
        print 'pr: {}'.format(self.pr(v, with_log_norm=True))

    # def init_message_passing(self):
    #     for f in self.factors:
    #         self.vp_table[f] = self.vp_node[f].table

    # def message_passing(self):
    #     self.init_message_passing()

    #     root = list(self.graph.vertices())[0]
    #     _, edges = self.traverse(root)

    #     self.clear_msgs()
    #     for e in reversed(edges):
    #         self.pass_msgs(e.target(), e.source())
    #     for e in edges:
    #         self.pass_msgs(e.source(), e.target())

    #     logger.debug('Message Passing Done.')

# # computing partition function for this specific instance of message passing
    #     _, self.Z = self.pr(self.variables[0], with_log_norm=True)
    #     # self.Z = pr.sum()
    #     self.logZ = np.log(self.Z)

    #     logger.debug('Z: %s', self.Z)
    #     logger.debug('logZ: %s', self.logZ)

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
        algo.log_messages(self, ('max-product',))

        v = self.variables[0]
        e = v.out_edges().next()
        vf_lnc = self.ep_mp_msg_vf_lnc[e]
        fv_lnc = self.ep_mp_msg_fv_lnc[e]
        return math.exp(vf_lnc + fv_lnc) * ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).max()

    def argmax(self):
        algo.log_messages(self, ('max-product',))
        return [ ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).argmax() for e in self.iter_first_edge() ]

    def iter_first_edge(self):
        for v in self.variables:
            yield v.out_edges().next()

    # @property
    # def feats(self):
    #     return self._feats

    # @property
    # def fdesc(self):
    #     return self._fdesc

    # @feats.setter
    # def feats(self, value):
    #     if isinstance(value, Feats):
    #         self._fdesc = value
    #         self._feats = np.empty(value._nfeats)
    #     elif isinstance(value, np.ndarray):
    #         if self.nfeats == value.size:
    #             self._feats[:] = value.ravel()
    #         else:
    #             self._feats = value.ravel()
    #     else:
    #         raise NotImplementedError('feats.setter with object type {} not defined.'.format(type(value)))

    # @property
    # def nfeats(self):
    #     if isinstance(self._feats, np.ndarray):
    #         return self._feats.size
    #     else:
    #         return None

    # @nfeats.setter
    # def nfeats(self, value):
    #     if isinstance(value, int) and value > 0:
    #         self.feats = np.empty(value)
    #     else:
    #         raise Exception('nfeats.setter with object type {} not defined.').format(type(value))

    @property
    def feats(self):
        return self._feats
    
    @feats.setter
    def feats(self, value):
        if isinstance(value, np.ndarray):
            if self.nfeats == value.size:
                self._feats[:] = value.ravel()
            else:
                self._feats = value.ravel()
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
    
    def l(self):
        return np.prod([ self.vp_node[f].value() for f in self.factors ])

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
# TODO fix the following as well? it might have the same bug as the dnll case
        if data is None:
# Assume data is already set, 
            # return -np.log([ self.vp_node[f].l() for f in self.factors ]).sum()
            return self.logZ - np.log(self.l())

        logger.debug('Computing NLL')

        if params is not None:
            self.params = params

# TODO implement later today
        Feats.extend(data)

        X, Y = data['X'], data['Y']
        Y_kwlist = data['Y_kwlist']
        
        ndata = X.shape[0]

        nll = np.empty(ndata)
        logger.debug('Allocating nll array with shape %s', nll.shape)
        for i, x, y, y_kw in itt.izip(itt.count(), X, Y, Y_kwlist):
            self.feats = x
            self.values = None
            self.values = y_kw

            for f in self.factors:
                self.vp_node[f].make_table()

            algo.message_passing(self, 'sum-product')

            nll[i] = self.nll()
        logger.debug('NLL DONE: %s', nll.sum())
        return nll

    def dnll(self, data = None, params = None):
        if data is None:
# Assume data is already set, 
            return np.array([ self.vp_node[f].gradient() for f in self.factors ]).sum(axis=0)

        logger.debug('Computing DNLL')

        if params is not None:
            self.params = params

# TODO implement later today
        Feats.extend(data)

        X, Y = data['X'], data['Y']
        Y_kwlist = data['Y_kwlist']
        
        ndata = X.shape[0]

        dnll = np.empty((ndata, self.nparams))
        logger.debug('Allocating dnll array with shape %s', dnll.shape)
        for i, x, y, y_kw in itt.izip(itt.count(), X, Y, Y_kwlist):
            self.feats = x
            self.values = None
            self.values = y_kw
            for f in self.factors:
                self.vp_node[f].make_table()
            algo.message_passing(self, 'sum-product')

            dnll[i] = self.dnll()
        logger.debug('DNLL DONE: %s', dnll.shape)

        return dnll

    def train(self, data):
        self.params = 'random'

        def fun(params, data):
            cost = self.nll(data = data, params = params).sum() 
            reg = 1. * la.norm(params, ord=1)
            logger.debug('Objectve Function: %s ( %s + %s )', cost+reg, cost, reg)
            return cost + reg

        def jac(params, data):
            dcost = self.dnll(data = data, params = params).sum(axis=0)
            dreg = 1. * np.sign(params)
            logger.debug('Objectve Jacobian: %s ( %s + %s )', dcost+dreg, dcost, dreg)
            return dcost + dreg

        logger.info('BEGIN Optimization')
        res = opt.minimize(
            fun  = fun,
            x0   = self.params,
            # method='nelder-mead',
            jac  = jac,
            args = (data,)
        )
        logger.info('END Optimization')
        logger.info('Training result: %s', res)
        # if not res.success:
        #     raise Exception('Failed to train!')
        self.params = res.x
