import math
import numpy as np
from scipy.misc import logsumexp

import logging
logger = logging.getLogger(__name__)

import pyfgraph.utils.log as log


def message_passing(fgraph, *args):
    # logger.debug('message_passing args %s', args)
    which = tuple(args)
    if not which:
        which = ('sum-product', 'max-product')

# traverse tree
    root = list(fgraph.graph.vertices())[0]
    # logger.debug('root is %s', fgraph.vp_name[root])
    boundary = [ root ]
    seen = []
    edges = []
    while boundary:
        node = boundary.pop()
        for neigh in node.out_neighbours():
            if neigh not in seen:
                edges.append(fgraph.graph.edge(node, neigh))
                boundary.append(neigh)
        seen.append(node)
    del boundary, seen

# initialize tables
    for f in fgraph.factors:
        # fgraph.vp_table[f] = fgraph.vp_node[f].table
        fgraph.vp_log_table[f] = fgraph.vp_node[f].log_table

# initialize messages
    for e in fgraph.graph.edges():
        if 'sum-product' in which:
            fgraph.ep_sp_log_msg_fv[e] = None
            fgraph.ep_sp_log_msg_vf[e] = None

            fgraph.logZ = None
        if 'max-product' in which:
            fgraph.ep_mp_log_msg_fv[e] = None
            fgraph.ep_mp_log_msg_vf[e] = None

    # logger.debug('I will traverse the edges in this order:')

    # for e in reversed(edges):
    #     logger.debug(' - %s -> %s', fgraph.vp_name[e.target()], fgraph.vp_name[e.source()])
    # for e in edges:
    #     logger.debug(' - %s -> %s', fgraph.vp_name[e.source()], fgraph.vp_name[e.target()])

# pass messages leaves->root
    for e in reversed(edges):
        _pass_msg(fgraph, e, e.target(), e.source(), which)

# pass messages root-leaves
    for e in edges:
        _pass_msg(fgraph, e, e.source(), e.target(), which)

    log.log_messages(fgraph, which)

    if 'sum-product' in which:
        fgraph.logZ = logsumexp(fgraph.log_pr(fgraph.variables[0]))
        # logger.debug('logZ: %s', fgraph.logZ)

def _pass_msg(fgraph, e, s, t, which):
    # log.log_messages(fgraph, which)

    snode = fgraph.vp_node[s]
    tnode = fgraph.vp_node[t]
    # logger.debug('passing %s -> %s', snode, tnode)

    s_vtype = fgraph.vp_type[s]
    if s_vtype == 'variable':
        arity = fgraph.vp_arity[s]
        sp_out_log_msg = np.zeros(arity)
        mp_out_log_msg = np.zeros(arity)

        # logger.debug('   === V -> F === ')
        for neigh in filter(lambda neigh: neigh != t, s.out_neighbours()):
            neigh_edge = fgraph.graph.edge(neigh, s)
            if 'sum-product' in which:
                sp_in_log_msg = fgraph.ep_sp_log_msg_fv[neigh_edge]
                # logger.debug(' * sp_in_log_msg: %s', sp_in_log_msg)
                sp_out_log_msg += sp_in_log_msg

            if 'max-product' in which:
                mp_in_log_msg = fgraph.ep_mp_log_msg_fv[neigh_edge]
                # logger.debug(' * mp_in_log_msg: %s', mp_in_log_msg)
                mp_out_log_msg += mp_in_log_msg

        if 'sum-product' in which:
            # logger.debug(' * sp_out_log_msg: %s', sp_out_log_msg)
            fgraph.ep_sp_log_msg_vf[e] = sp_out_log_msg
        if 'max-product' in which:
            # logger.debug(' * mp_out_log_msg: %s', mp_out_log_msg)
            fgraph.ep_mp_log_msg_vf[e] = mp_out_log_msg
    elif s_vtype  == 'factor':
        nname = fgraph.vp_name[t]
        msgs = {}
        if 'sum-product' in which:
            msgs['sum-product'] = { nname: np.zeros(fgraph.vp_arity[t]) }
        if 'max-product' in which:
            msgs['max-product'] = { nname: np.zeros(fgraph.vp_arity[t]) }

        # logger.debug('   === F -> V === ')
        for neigh in filter(lambda neigh: neigh != t, s.out_neighbours()):
            nname = fgraph.vp_name[neigh]
            neigh_edge = fgraph.graph.edge(neigh, s)
            if 'sum-product' in which:
                sp_in_log_msg = fgraph.ep_sp_log_msg_vf[neigh_edge]
                # logger.debug(' * sp_in_log_msg:     %s', sp_in_log_msg)
                msgs['sum-product'][nname] = sp_in_log_msg
            if 'max-product' in which:
                mp_in_log_msg = fgraph.ep_mp_log_msg_vf[neigh_edge]
                # logger.debug(' * mp_in_log_msg:     %s', mp_in_log_msg)
                msgs['max-product'][nname] = mp_in_log_msg

        if 'sum-product' in which:
            msgs['sum-product'] = [ msgs['sum-product'][n] for n in fgraph.vp_table_inputs[s] ]
        if 'max-product' in which:
            msgs['max-product'] = [ msgs['max-product'][n] for n in fgraph.vp_table_inputs[s] ]

        # logger.debug(' * log_factor: %s', fgraph.vp_log_table[s])

        prod = {}
        if 'sum-product' in which:
            prod['sum-product'] = fgraph.vp_log_table[s] + reduce(np.add, np.ix_(*msgs['sum-product']))
        if 'max-product' in which:
            prod['max-product'] = fgraph.vp_log_table[s] + reduce(np.add, np.ix_(*msgs['max-product']))

        axis = fgraph.vp_table_inputs[s].index(fgraph.vp_name[t])
        negaxis = tuple( a for a in xrange(len(fgraph.vp_log_table[s].shape)) if a != axis )

        if 'sum-product' in which:
            sp_out_log_msg = logsumexp(prod['sum-product'], axis=negaxis)
            # logger.debug(' * sp_out_log_msg:     %s', sp_out_log_msg)
            fgraph.ep_sp_log_msg_fv[e] = sp_out_log_msg
        if 'max-product' in which:
            mp_out_log_msg = prod['max-product'].max(axis=negaxis)
            # logger.debug(' * mp_out_log_msg:     %s', mp_out_log_msg)
            fgraph.ep_mp_log_msg_fv[e] = mp_out_log_msg
    else:
        raise Exception('variable type error: {}'.format(s_vtype))

def check_message_passing(fgraph):
    # print 'Checking the message passing. For each variable, all the rows should be the same.'
    msgs = tuple(
        np.vstack((
            np.sum([ fgraph.ep_sp_log_msg_fv[e] for e in v.out_edges() ], axis=0),
            np.array([
                fgraph.ep_sp_log_msg_vf[e] + fgraph.ep_sp_log_msg_fv[e]
                    for e, v in ((e, v) for e in v.out_edges())
            ])
        )) for v in fgraph.variables )

    return msgs

    # for v in self.variables:
    #     print 'Variable', self.vp_name[v]
    #     print np.sum([ self.ep_sp_log_msg_fv[e] for e in v.out_edges() ], axis = 0)
    #     for e in v.out_edges():
    #         msg_vf = self.ep_sp_log_msg_vf[e]
    #         msg_fv = self.ep_sp_log_msg_fv[e]
    #         print msg_vf + msg_fv

