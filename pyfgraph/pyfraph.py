import sys, warnings

warnings.filterwarnings('error')

from graph_tool.all import *
from graph_tool.util import find_vertex

import scipy.optimize as opt

import itertools

import numpy as np
import numpy.random as rnd
import numpy.linalg as la

import logging
import nptk, log

logger = logging.getLogger()

# NB: this does not in itself contain the parameters. It just represents the shared parameters
# Q: where is the actual parameter array? in the fg.
class Params(object):
    tot_nparams = 0
    def __init__(self, n):
        self.n = n
        self.pslice = slice(Params.tot_nparams, Params.tot_nparams+n)
        Params.tot_nparams += n

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


class FactorGraph(object):
    def __init__(self):
        self.graph = Graph(directed=False)
        self.variables = []
        self.factors = []
        self.Z = None
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
        self.ep_sp_msg_fv_nc = self.graph.new_edge_property("double")
        self.ep_sp_msg_vf = self.graph.new_edge_property("object")
        self.ep_sp_msg_vf_nc = self.graph.new_edge_property("double")

        self.ep_mp_msg_fv = self.graph.new_edge_property("object")
        self.ep_mp_msg_fv_nc = self.graph.new_edge_property("double")
        self.ep_mp_msg_vf = self.graph.new_edge_property("object")
        self.ep_mp_msg_vf_nc = self.graph.new_edge_property("double")

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
            self.nparams = Params.tot_nparams
            self.params = np.empty(self.nparams)
            # i = 0
            # for p in pset:
            #     p.pslice = slice(i, i+p.n)
            #     i += p.n

            for f in self.factors:
                node = self.vp_node[f]
                node.pslice = node.params.pslice
                node.params = self.params[node.pslice]
                node.params_tab = node.params.view()
                node.params_tab.shape = node.pshape

# Just testing that all the views are set correctly
            # self.params[:] = 10
            # for f in self.factors:
            #     node = self.vp_node[f]
            #     print 'node: {}'.format(node)
            #     print ' * pslice: {}'.format(node.pslice)
            #     print ' * params: {}'.format(node.params)

            # self.params[:] = np.arange(self.params.size)
            # for f in self.factors:
            #     node = self.vp_node[f]
            #     print 'node: {}'.format(node)
            #     print ' * pslice: {}'.format(node.pslice)
            #     print ' * params: {}'.format(node.params)
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

    def set_values(self, **kwargs):
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

    def clear_msgs(self):
        for e in self.graph.edges():
            self.ep_sp_msg_fv[e] = None
            self.ep_sp_msg_vf[e] = None

            self.ep_sp_msg_fv_nc[e] = 0.
            self.ep_sp_msg_vf_nc[e] = 0.

            self.ep_mp_msg_fv[e] = None
            self.ep_mp_msg_vf[e] = None

            self.ep_mp_msg_fv_nc[e] = 0.
            self.ep_mp_msg_vf_nc[e] = 0.
            
            self.Z = None
            self.logZ = None

    def pass_msgs(self, s, t):
        snode = self.vp_node[s]
        tnode = self.vp_node[t]
        logger.debug('passing %s -> %s', snode, tnode)

        s_vtype = self.vp_type[s]
        if s_vtype == 'variable':
            arity = self.vp_arity[s]
            sp_msg, sp_msg_nc = np.ones(arity), 1
            mp_msg, mp_msg_nc = np.ones(arity), 1

            logger.debug(' * sp_msg:    %s', sp_msg)
            logger.debug(' * sp_msg_nc: %s', sp_msg_nc)
            logger.debug(' * mp_msg:    %s', mp_msg)
            logger.debug(' * mp_msg_nc: %s', mp_msg_nc)
            neighbours = list(s.out_neighbours())
            neighbours.remove(t)
            for neigh in neighbours:
                e = self.graph.edge(neigh, s)
                sp_msg    *= self.ep_sp_msg_fv[e]
                sp_msg_nc *= self.ep_sp_msg_fv_nc[e]
                mp_msg    *= self.ep_mp_msg_fv[e]
                mp_msg_nc *= self.ep_mp_msg_fv_nc[e]
                logger.debug('   -------')
                logger.debug(' * sp_msg:    %s', sp_msg)
                logger.debug(' * sp_msg_nc: %s', sp_msg_nc)
                logger.debug(' * mp_msg:    %s', mp_msg)
                logger.debug(' * mp_msg_nc: %s', mp_msg_nc)

            e = self.graph.edge(s, t)

            sp_msg_nc *= sp_msg.sum()
            self.ep_sp_msg_vf[e] = sp_msg/sp_msg.sum()
            self.ep_sp_msg_vf_nc[e] = sp_msg_nc

            mp_msg_nc *= mp_msg.sum()
            self.ep_mp_msg_vf[e] = mp_msg/mp_msg.sum()
            self.ep_mp_msg_vf_nc[e] = mp_msg_nc
        elif s_vtype  == 'factor':
            nname = self.vp_name[t]
            msgs = { nname: np.ones(self.vp_arity[t]) }
            msg_nc = 1
            logger.debug(' * in_msg:    %s', msgs[nname])
            logger.debug(' * in_msg_nc: %s', msg_nc)

            neighbours = list(s.out_neighbours())
            # neighbours.remove(t)
            for neigh in filter(lambda neigh: neigh != t, neighbours):
                nname = self.vp_name[neigh]
                e = self.graph.edge(neigh, s)
                msgs[nname] = self.ep_mp_msg_vf[e]
                msg_nc *= self.ep_mp_msg_vf_nc[e]
                logger.debug(' * in_msg:    %s', self.ep_mp_msg_vf[e])
                logger.debug(' * in_msg_nc: %s', self.ep_mp_msg_vf_nc[e])

            msgs = [ msgs[n] for n in self.vp_table_inputs[s] ]

            logger.debug(' * factor: %s', self.vp_table[s])

            prod = self.vp_table[s] * reduce(np.multiply, np.ix_(*msgs))
            axis = self.vp_table_inputs[s].index(self.vp_name[t])
            negaxis = tuple(filter(lambda a: a!= axis, range(len(neighbours))))
            sp_msg = prod.sum(axis = negaxis)
            mp_msg = prod.max(axis = negaxis)

            e = self.graph.edge(s, t)

            sp_msg_sum = sp_msg.sum()
            self.ep_sp_msg_fv_nc[e] = msg_nc * sp_msg_sum
            self.ep_sp_msg_fv[e] = sp_msg/sp_msg_sum

            mp_msg_sum = mp_msg.sum()
            self.ep_mp_msg_fv_nc[e] = msg_nc * mp_msg_sum
            self.ep_mp_msg_fv[e] = mp_msg/mp_msg_sum

            logger.debug(' * out_msg_sp:    %s', self.ep_sp_msg_fv[e])
            logger.debug(' * out_msg_sp_nc: %s', self.ep_sp_msg_fv_nc[e])

            logger.debug(' * out_msg_mp:    %s', self.ep_mp_msg_fv[e])
            logger.debug(' * out_msg_mp_nc: %s', self.ep_mp_msg_fv_nc[e])
        else:
            raise Exception('variable type error: {}'.format(s_vtype))

    def traverse(self, root):
        boundary = [ root ]
        seen = []
        edges = []
        while boundary:
            node = boundary.pop()
            for neigh in node.out_neighbours():
                if neigh not in seen:
                    edges.append(self.graph.edge(node, neigh))
                    boundary.append(neigh)
            seen.append(node)
        return seen, edges

    def print_marginals(self, v):
        print 'Marginal Probabilities Variables'
        print 'v: ', v
        print 'variable: {}'.format(self.vp_name[v])
        print 'pr: {}'.format(self.pr(v, with_l1_norm=True))

    def init_message_passing(self):
        for f in self.factors:
            self.vp_table[f] = self.vp_node[f].table

    def message_passing(self):
        self.init_message_passing()

        root = list(self.graph.vertices())[0]
        _, edges = self.traverse(root)

        self.clear_msgs()
        for e in reversed(edges):
            self.pass_msgs(e.target(), e.source())
        for e in edges:
            self.pass_msgs(e.source(), e.target())

        logger.debug('Message Passing Done.')

# computing partition function for this specific instance of message passing
        pr, self.Z = self.pr(self.variables[0], with_l1_norm=True)
        # self.Z = pr.sum()
        self.logZ = np.log(self.Z)

        logger.debug('Z: %s', self.Z)
        logger.debug('logZ: %s', self.logZ)

    def check_message_passing(self):
        for v in self.variables:
            print 'Variable', self.vp_name[v]
            print np.array([ self.ep_sp_msg_fv[e] for e in v.out_edges() ]).prod(axis=0)
            for e in v.out_edges():
                msg_vf = self.ep_sp_msg_vf[e]
                msg_fv = self.ep_sp_msg_fv[e]
                print msg_vf * msg_fv

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

    def pr(self, vertex, with_l1_norm=False):
        s_vtype = self.vp_type[vertex]
        if s_vtype == 'variable':
            e = vertex.out_edges().next()
            pr = self.ep_sp_msg_vf[e] * self.ep_sp_msg_fv[e]
            prs = pr.sum()

            if with_l1_norm:
                nc = self.ep_sp_msg_vf_nc[e] * self.ep_sp_msg_fv_nc[e] * prs
        elif s_vtype == 'factor':
            msgs = { self.vp_name[e.target()]: self.ep_sp_msg_vf[e] for e in vertex.out_edges() }
            msgs = [ msgs[n] for n in self.vp_table_inputs[vertex] ]
            pr = self.vp_table[vertex] * reduce(np.multiply, np.ix_(*msgs))
            prs = pr.sum()

            if with_l1_norm:
                nc = np.array([ self.ep_sp_msg_vf_nc[e] for e in vertex.out_edges() ]).prod() * prs
        else:
            raise Exception('variable type error: {}'.format(s_vtype))

        logger.debug('pr: %s'.format(pr))
        pr /= prs
        logger.debug('pr: %s'.format(pr))
        return (pr, nc) if with_l1_norm else pr

    def max(self):
        v = self.variables[0]
        e = v.out_edges().next()
        return ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).max()

    def argmax(self):
        return [
            ( self.ep_mp_msg_vf[e] * self.ep_mp_msg_fv[e] ).argmax() for e in self.iter_first_edge()
        ]

    def iter_first_edge(self):
        for v in self.variables:
            yield v.out_edges().next()

# TODO revolutionize this. No need to keep on concatenating stuff and so on.....

    def getParams(self):
        # return np.concatenate(tuple( self.vp_node[f].getParams() for f in self.factors ))
        return self.params

    def setParams(self, params):
        self.params[:] = params
        # pdims = np.array([ self.vp_node[f].nfeats * np.prod(self.vp_node[f].arity) for f in self.factors ])
        # params = nptk.split_as(params, pdims)
        # for f, p in zip(self.factors, params):
        #     self.vp_node[f].setParams(p)

# TODO this is supposedly, if the features are different for each factor
        # pdims = np.array([ self.vp_node[f].pdim for f in self.factors ])
        # params = split_as(params, pdims)
        # for f, p in zip(self.factors, params):
        #     self.vp_node[f].setParams(p)

    def l(self):
        return np.prod([ self.vp_node[f].value() for f in self.factors ])

    def viterbi(self, data = None, params = None):
        if params is not None:
            self.setParams(params)

        vit = []
        for d in data:
            self.clear_values()
            for f in self.factors:
                self.vp_node[f].setFeats(d['phi_'])
                self.vp_node[f].makeTable()

            self.clear_msgs()
            self.message_passing()

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

        nll = np.empty(len(data))
        for i, d in enumerate(data):
            self.clear_values()
            self.set_values(**d)

            for f in self.factors:
                self.vp_node[f].setFeats(d['phi_'])
                self.vp_node[f].makeTable()

            self.clear_msgs()
            self.message_passing()

            nll[i] = self.nll()
        logger.debug('NLL DONE: %s'.format(nll.shape))
        return nll

    def dnll(self, data = None, params = None):
        if data is None:
# Assume data is already set, 
            # return np.concatenate(tuple(self.vp_node[f].gradient().ravel() for f in self.factors))
            return np.array([ self.vp_node[f].gradient() for f in self.factors ]).sum(axis=0)

        logger.debug('Computing DNLL')

        if params is not None:
            self.setParams(params)

        dnll = np.empty((len(data), self.nparams))
        logger.debug('Allocating dnll array with shape %s', dnll.shape)
        for i, d in enumerate(data):
            self.clear_values()
            self.set_values(**d)

            for f in self.factors:
                self.vp_node[f].setFeats(d['phi_'])
                self.vp_node[f].makeTable()

            self.clear_msgs()
            self.message_passing()

            dnll[i] = self.dnll()
        logger.debug('DNLL DONE: {}'.format(dnll.shape))

        return dnll

    def initFactorDims(self, d):
        nfeats = d['phi_'].size
        for f in self.factors:
            self.vp_node[f].nfeats = nfeats

    def train(self, data):
        self.initFactorDims(data[0])
        for f in self.factors:
            self.vp_node[f].setParams('random')

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
            # callback = lambda params: logger.info(' * params: %s', params)
        )
        logger.info('END   Optimization')
        logger.info('Training result: {}'.format(res))
        # if not res.success:
        #     raise Exception('Failed to train!')
        self.setParams(res.x)

def testFactorGraph():
    fg = FactorGraph()

    RIGHT = fg.add(Variable, 'RIGHT', 3)
    TOP = fg.add(Variable, 'TOP', 3)
    RED = fg.add(Variable, 'RED', 3)

    f_RIGHT = fg.add(Factor, 'f_RIGHT', RIGHT)
    f_TOP = fg.add(Factor, 'f_TOP', TOP)
    f_RED = fg.add(Factor, 'f_RED', RED)
    f_star = fg.add(Factor, 'f_star', RIGHT, TOP, RED)

    f_RIGHT.table = np.array([1, 0, 1])
    f_TOP.table = np.array([1, 1, 0])
    f_RED.table = np.array([1, 1, 1])
    f_star.table = np.zeros((3, 3, 3))
    f_star.table[0, 0, 0] = 1
    f_star.table[0] = 1

    fg.message_passing()
    fg.check_message_passing()

    print 'argmax: {}'.format(fg.argmax())
    print 'max: {}'.format(fg.max())

    print 'partition: '
    print fg.Z
    print fg.logZ

def testCRF():
    fg = FactorGraph()

    RIGHT = fg.add(Variable, 'RIGHT', arity=2)
    TOP = fg.add(Variable, 'TOP', arity=2)
    QUART = fg.add(Variable, 'QUART', arity=2)

    f_RIGHT = fg.add(FFactor, 'f_RIGHT', RIGHT)
    f_TOP = fg.add(FFactor, 'f_TOP', TOP)
    f_QUART = fg.add(FFactor, 'f_QUART', QUART)
    f_star = fg.add(FFactor, 'f_star', RIGHT, TOP, QUART)

    n = 10
    X = rnd.randn(n, nfeats)
    Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)

    data = tuple( { 'phi_':     x,
                    'RIGHT': y[0],
                    'TOP':   y[1],
                    'QUART': y[2] } for x, y in zip(X, Y) )
    # f_RIGHT.setParams(1)
    # f_TOP.setParams(1)
    # f_QUART.setParams(1)
    # f_star.setParams(1)

    np.set_printoptions(threshold=10)

    print 'X'
    print X
    print 'Y'
    print Y

    print fg.train(data)

    print 'Training done:'

    print 'Training data:'
    nll = fg.nll(data=data)
    vit = fg.viterbi(data=data)
    for x, y, v, l in zip(X, Y, vit, nll):
        print x, y, v, l

    n = 100
    XX = rnd.randn(n, d)
    YY = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in XX ], dtype=int)
    # ZZ = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in XX ], dtype=int)
    data = tuple( { 'phi_':     x,
                    'RIGHT': y[0],
                    'TOP':   y[1],
                    'QUART': y[2] } for x, y in zip(XX, YY) )

    print 'other data:'
    nll = fg.nll(data=data)
    vit = fg.viterbi(data=data)
    for x, y, v, l in zip(XX, YY, vit, nll):
        print x, y, v, l

def testCRF_shared_params():
    fg = FactorGraph()

    RIGHT   = fg.add(Variable, 'RIGHT', arity=2)
    TOP     = fg.add(Variable, 'TOP',   arity=2)
    QUART   = fg.add(Variable, 'QUART', arity=2)

    # p_RIGHT = Params(n=4)
    # p_TOP   = Params(n=4)
    # p_QUART = Params(n=4)
    # p_star  = Params(n=16)

    # f_RIGHT = fg.add(FFactor, 'f_RIGHT',    RIGHT,                  p_RIGHT )
    # f_TOP   = fg.add(FFactor, 'f_TOP',      TOP,                    p_TOP   )
    # f_QUART = fg.add(FFactor, 'f_QUART',    QUART,                  p_QUART )
    # f_star  = fg.add(FFactor, 'f_star',     (RIGHT, TOP, QUART),    p_star  )

    p_var   = Params(n=4)
    p_star  = Params(n=16)

    f_RIGHT = fg.add(FFactor, 'f_RIGHT',    RIGHT,                  p_var   )
    f_TOP   = fg.add(FFactor, 'f_TOP',      TOP,                    p_var   )
    f_QUART = fg.add(FFactor, 'f_QUART',    QUART,                  p_var   )
    f_star  = fg.add(FFactor, 'f_star',     (RIGHT, TOP, QUART),    p_star  )

    fg.make(done=True)

    n = 10
    d = 2
    X = rnd.randn(n, d)
    Y = np.array([ [ x[0]>=0, x[1]>=0, x[0]>=x[1] ] for x in X ], dtype=int)

    data = tuple( { 'phi_':     x,
                    'RIGHT': y[0],
                    'TOP':   y[1],
                    'QUART': y[2] } for x, y in zip(X, Y) )
    # f_RIGHT.setParams(1)
    # f_TOP.setParams(1)
    # f_QUART.setParams(1)
    # f_star.setParams(1)

    np.set_printoptions(threshold=10)

    print 'X:'
    print X
    print 'Y:'
    print Y

    fg.train(data)

    logger.info('Training done.')

    logger.info('Results:')
    nll = fg.nll(data=data)
    vit = fg.viterbi(data=data)
    for x, y, v, l in zip(X, Y, vit, nll):
        logger.info('%s %s %s %s', x, y, v, l)

if __name__ == '__main__':
    rnd.seed(1)
    # testFactorGraph()

    # testCRF()
    testCRF_shared_params()
