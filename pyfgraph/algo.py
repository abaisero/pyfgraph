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
        fgraph.vp_table[f] = fgraph.vp_node[f].table
        fgraph.vp_nl_table[f] = fgraph.vp_node[f].nl_table

# initialize messages
    for e in fgraph.graph.edges():
        if 'sum-product' in which:
            fgraph.ep_sp_msg_fv[e] = None
            fgraph.ep_sp_msg_vf[e] = None

            fgraph.ep_sp_msg_fv_lnc[e] = -np.inf
            fgraph.ep_sp_msg_vf_lnc[e] = -np.inf

            fgraph.logZ = None
        if 'max-product' in which:
            fgraph.ep_mp_msg_fv[e] = None
            fgraph.ep_mp_msg_vf[e] = None

            fgraph.ep_mp_msg_fv_lnc[e] = -np.inf
            fgraph.ep_mp_msg_vf_lnc[e] = -np.inf
    
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
        _, fgraph.logZ = fgraph.pr(fgraph.variables[0], with_log_norm=True)
        # logger.debug('logZ: %s', fgraph.logZ)

def _pass_msg(fgraph, e, s, t, which):
    log.log_messages(fgraph, which)

    snode = fgraph.vp_node[s]
    tnode = fgraph.vp_node[t]
    # logger.debug('passing %s -> %s', snode, tnode)

    s_vtype = fgraph.vp_type[s]
    if s_vtype == 'variable':
        arity = fgraph.vp_arity[s]
        sp_msg, sp_msg_lnc = np.ones(arity), 0
        mp_msg, mp_msg_lnc = np.ones(arity), 0

        for neigh in filter(lambda neigh: neigh != t, s.out_neighbours()):
            neigh_edge = fgraph.graph.edge(neigh, s)
            # logger.debug('   -------')
            if 'sum-product' in which:
                sp_in_msg = fgraph.ep_sp_msg_fv[neigh_edge]
                sp_in_msg_lnc = fgraph.ep_sp_msg_fv_lnc[neigh_edge]

                # logger.debug(' * sp_in_msg:     %s', sp_in_msg)
                # logger.debug(' * sp_in_msg_lnc: %s', sp_in_msg_lnc)

                sp_msg     *= sp_in_msg
                sp_msg_lnc += sp_in_msg_lnc
            if 'max-product' in which:
                mp_in_msg = fgraph.ep_mp_msg_fv[neigh_edge]
                mp_in_msg_lnc = fgraph.ep_mp_msg_fv_lnc[neigh_edge]

                # logger.debug(' * mp_in_msg:     %s', mp_in_msg)
                # logger.debug(' * mp_in_msg_lnc: %s', mp_in_msg_lnc)

                mp_msg     *= mp_in_msg
                mp_msg_lnc += mp_in_msg_lnc

        if 'sum-product' in which:
            sp_msg_sum = sp_msg.sum()
            sp_out_msg = sp_msg/sp_msg_sum
            sp_out_msg_lnc = sp_msg_lnc + math.log(sp_msg_sum)

            # logger.debug(' * sp_out_msg:     %s', sp_out_msg)
            # logger.debug(' * sp_out_msg_lnc: %s', sp_out_msg_lnc)

            fgraph.ep_sp_msg_vf[e] = sp_out_msg
            fgraph.ep_sp_msg_vf_lnc[e] = sp_out_msg_lnc
        if 'max-product' in which:
            mp_msg_sum = mp_msg.sum()
            mp_out_msg = mp_msg/mp_msg_sum
            mp_out_msg_lnc = mp_msg_lnc + math.log(mp_msg_sum)

            # logger.debug(' * mp_out_msg:     %s', mp_out_msg)
            # logger.debug(' * mp_out_msg_lnc: %s', mp_out_msg_lnc)

            fgraph.ep_mp_msg_vf[e] = mp_out_msg
            fgraph.ep_mp_msg_vf_lnc[e] = mp_out_msg_lnc
    elif s_vtype  == 'factor':
        nname = fgraph.vp_name[t]
        msgs, msgs_lnc = {}, {}
        if 'sum-product' in which:
            msgs['sum-product'] = { nname: np.ones(fgraph.vp_arity[t]) }
            msgs_lnc['sum-product'] = 0
        if 'max-product' in which:
            msgs['max-product'] = { nname: np.ones(fgraph.vp_arity[t]) }
            msgs_lnc['max-product'] = 0

        for neigh in filter(lambda neigh: neigh != t, s.out_neighbours()):
            nname = fgraph.vp_name[neigh]
            neigh_edge = fgraph.graph.edge(neigh, s)
            if 'sum-product' in which:
                sp_in_msg = fgraph.ep_sp_msg_vf[neigh_edge]
                sp_in_msg_lnc = fgraph.ep_sp_msg_vf_lnc[neigh_edge]

                # logger.debug(' * sp_in_msg:     %s', sp_in_msg)
                # logger.debug(' * sp_in_msg_lnc: %s', sp_in_msg_lnc)

                msgs['sum-product'][nname] = sp_in_msg
                msgs_lnc['sum-product'] += sp_in_msg_lnc
            if 'max-product' in which:
                mp_in_msg = fgraph.ep_mp_msg_vf[neigh_edge]
                mp_in_msg_lnc = fgraph.ep_mp_msg_vf_lnc[neigh_edge]

                # logger.debug(' * mp_in_msg:     %s', mp_in_msg)
                # logger.debug(' * mp_in_msg_lnc: %s', mp_in_msg_lnc)

                msgs['max-product'][nname] = mp_in_msg
                msgs_lnc['max-product'] += mp_in_msg_lnc

        if 'sum-product' in which:
            msgs['sum-product'] = [ msgs['sum-product'][n] for n in fgraph.vp_table_inputs[s] ]
        if 'max-product' in which:
            msgs['max-product'] = [ msgs['max-product'][n] for n in fgraph.vp_table_inputs[s] ]

        # logger.debug(' * factor: %s', fgraph.vp_table[s])

        prod = {}
# TODO change into vp_nl_table
        if 'sum-product' in which:
            prod['sum-product'] = fgraph.vp_table[s] * reduce(np.multiply, np.ix_(*msgs['sum-product']))
        if 'max-product' in which:
            prod['max-product'] = fgraph.vp_table[s] * reduce(np.multiply, np.ix_(*msgs['max-product']))

        axis = fgraph.vp_table_inputs[s].index(fgraph.vp_name[t])
        negaxis = tuple( a for a in xrange(len(fgraph.vp_table[s].shape)) if a != axis )

        if 'sum-product' in which:
            sp_msg = prod['sum-product'].sum(axis = negaxis)
            sp_msg_sum = sp_msg.sum()
            sp_out_msg = sp_msg/sp_msg_sum
            sp_out_msg_lnc = msgs_lnc['sum-product'] + math.log(sp_msg_sum)

            # logger.debug(' * sp_out_msg:     %s', sp_out_msg)
            # logger.debug(' * sp_out_msg_lnc: %s', sp_out_msg_lnc)

            fgraph.ep_sp_msg_fv[e] = sp_out_msg
            fgraph.ep_sp_msg_fv_lnc[e] = sp_out_msg_lnc
        if 'max-product' in which:
            mp_msg = prod['max-product'].max(axis = negaxis)
            mp_msg_sum = mp_msg.sum()
            mp_out_msg = mp_msg/mp_msg_sum
            mp_out_msg_lnc = msgs_lnc['max-product'] + math.log(mp_msg_sum)

            # logger.debug(' * mp_out_msg:     %s', mp_out_msg)
            # logger.debug(' * mp_out_msg_lnc: %s', mp_out_msg_lnc)

            fgraph.ep_mp_msg_fv[e] = mp_out_msg
            fgraph.ep_mp_msg_fv_lnc[e] = mp_out_msg_lnc
    else:
        raise Exception('variable type error: {}'.format(s_vtype))

