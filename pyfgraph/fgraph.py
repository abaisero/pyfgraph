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

# from params import Params
from nodes import Node, Variable, Factor, ParamFactor, FeatFactor, FunFactor
from algo import message_passing

import logging
logger = logging.getLogger(__name__)

class FactorGraph(object):
    def __init__(self):
        self.graph = Graph(directed=False)
        self.variables = []
        self.factors = []
        self.logZ = None

        self.make_done = False

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

    def add(self, cls, *args, **kwargs):
        if self.make_done:
            raise Exception('MakeGraph is not done')

        vertex = self.graph.add_vertex()
        if not issubclass(cls, Node):
            raise Exception('what are you doing..')

        c = cls(vertex, *args, **kwargs)
        c.graph = self
        self.vp_node[vertex] = c
        self.vp_name[vertex] = c.name

        if issubclass(cls, Variable):
            self.vp_type[vertex] = "variable"
            self.vp_shape[vertex] = "circle"
            self.vp_color[vertex] = "white"
            self.vp_size[vertex] = 50

# variable-specific properties
            self.vp_arity[vertex] = c.arity
            self.variables.append(vertex)
        else:
            self.vp_type[vertex] = "factor"
            self.vp_shape[vertex] = "square"
            self.vp_color[vertex] = "black"
            self.vp_size[vertex] = 30

# factor-specific properties
            self.vp_table_inputs[vertex] = [ self.vp_name[variable.vertex] for variable in c.variables ]
            self.factors.append(vertex)

# adding all edges of this factor
            for variable in c.variables:
                self.graph.add_edge(vertex, variable.vertex)

        return c

    def make(self, done=False):
        if done:
            self.nparams = ParamFactor.nparams
            self.params = np.empty(self.nparams)

            for f in self.factors:
                self.vp_node[f].make_params(self.params)
        else:
            pass
# TODO set other variables
        self.make_done = done

    def plot(self):
        graph_draw(
            self.graph,
            vertex_text=self.vp_name,
            vertex_shape=self.vp_shape,
            vertex_color="black",
            vertex_fill_color=self.vp_color,
            vertex_size=self.vp_size
        )

    def set_values(self, clear=False, **kwargs):
        if clear:
            self.clear_values()
        for key, value in kwargs.iteritems():
            vertices = find_vertex(self.graph, self.vp_name, key)
            if len(vertices) == 1:
                v = vertices[0]
                self.vp_node[v].value = value
            elif len(vertices) > 1:
                raise Exception('This should not happen.')

    def clear_values(self):
        for v in self.variables:
            self.vp_node[v].value = None

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

        logger.debug('pr: %s'.format(pr))
        pr /= prs
        logger.debug('pr: %s'.format(pr))
        return (pr, lnc) if with_log_norm else pr

    def max(self):
        v = self.variables[0]
        e = v.out_edges().next()
        vf_nc = math.exp(self.ep_mp_msg_vf_lnc[e])
        fv_nc = math.exp(self.ep_mp_msg_fv_lnc[e])
        return vf_nc * fv_nc * ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).max()
        # return ( vf_nc * self.ep_mp_msg_vf[e] * fv_nc * self.ep_mp_msg_fv[e] ).max()

    def argmax(self):
        return [
            ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).argmax() for e in self.iter_first_edge()
        ]

    def iter_first_edge(self):
        for v in self.variables:
            yield v.out_edges().next()

# TODO revolutionize this. No need to keep on concatenating stuff and so on.....

    def getParams(self):
        return self.params

    def setParams(self, params):
        if isinstance(params, np.ndarray):
            self.params[:] = params.ravel()
        elif params == 'random':
            self.params[:] = .1 * rnd.randn(self.params.size)
        else:
            raise NotImplementedError('setParams with {} not done yet').format(params)
        self.nparams = self.params.size

    def l(self):
        return np.prod([ self.vp_node[f].value() for f in self.factors ])

    def viterbi(self, data = None, params = None):
        if params is not None:
            self.setParams(params)

        vit = []
        for x in data['X']:
            self.clear_values()
            for f in self.factors:
                self.vp_node[f].make_table(phi=x, y_kw=None)

            # self.clear_msgs()
            message_passing(self, 'max-product')
            # self.message_passing()

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
            self.setParams(params)

        X, Y = data['X'], data['Y']
        Y_kwlist = data['Y_kwlist']
        
        ndata = X.shape[0]

        nll = np.empty(ndata)
        logger.debug('Allocating nll array with shape %s', nll.shape)
        for i, x, y, y_kw in itt.izip(itt.count(), X, Y, Y_kwlist):
            self.set_values(clear=True, **y_kw)

            for f in self.factors:
                self.vp_node[f].make_table(phi=x, y_kw=y_kw)

            # self.clear_msgs()
            # self.message_passing()
            message_passing(self, 'sum-product')

            nll[i] = self.nll()
        logger.debug('NLL DONE: %s', nll.shape)
        return nll

    def dnll(self, data = None, params = None):
        if data is None:
# Assume data is already set, 
            return np.array([ self.vp_node[f].gradient() for f in self.factors ]).sum(axis=0)

        logger.debug('Computing DNLL')

        if params is not None:
            self.setParams(params)


        X, Y = data['X'], data['Y']
        Y_kwlist = data['Y_kwlist']
        
        ndata = X.shape[0]

        dnll = np.empty((ndata, self.nparams))
        logger.debug('Allocating dnll array with shape %s', dnll.shape)
        for i, x, y, y_kw in itt.izip(itt.count(), X, Y, Y_kwlist):
            self.set_values(clear=True, **y_kw)

            for f in self.factors:
                self.vp_node[f].make_table(phi=x, y_kw=y_kw)

            # self.clear_msgs()
            # self.message_passing()
            message_passing(self, 'sum-product')

            dnll[i] = self.dnll()
        logger.debug('DNLL DONE: %s', dnll.shape)

        return dnll

    def initFactorDims(self, data):
        nfeats = data['X'].shape[1]
        for f in self.factors:
            self.vp_node[f].nfeats = nfeats

    def train(self, data):
        self.initFactorDims(data)
        self.setParams('random')

        def fun(params):
            cost = self.nll(data = data, params = params).sum() 
            reg = 1. * la.norm(params, ord=1)
            logger.debug('fun cost: (shape %s) %s', cost.shape, cost)
            logger.debug('fun reg:  (shape %s) %s', reg.shape, reg)
            return cost + reg

        def jac(params):
            dcost = self.dnll(data = data, params = params).sum(axis=0)
            dreg = 1. * np.sign(params)
            logger.debug('jac dcost: (shape %s) %s', dcost.shape, dcost)
            logger.debug('jac dreg:  (shape %s) %s', dreg.shape, dreg)
            return dcost + dreg

        logger.info('BEGIN Optimization')
        x0 = self.getParams()
        res = opt.minimize(
            fun      = fun,
            x0       = self.getParams(),
            jac      = jac,
        )
        logger.info('END   Optimization')
        logger.info('Training result: {}'.format(res))
        # if not res.success:
        #     raise Exception('Failed to train!')
        self.setParams(res.x)
