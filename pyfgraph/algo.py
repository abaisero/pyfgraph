import math
import numpy as np

import logging
logger = logging.getLogger(__name__)

def message_passing(fgraph, *args):
    logger.debug('message_passing args %s', args)
    which = tuple(args)
    if not which:
        which = ('sum-product', 'max-product')

# traverse tree
    root = list(fgraph.graph.vertices())[0]
    logger.debug('root is %s', fgraph.vp_name[root])
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
    
    logger.debug('I will traverse the edges in this order:')
    for e in reversed(edges):
        logger.debug(' - %s -> %s', fgraph.vp_name[e.target()], fgraph.vp_name[e.source()])
    for e in edges:
        logger.debug(' - %s -> %s', fgraph.vp_name[e.source()], fgraph.vp_name[e.target()])

# pass messages leaves->root
    for e in reversed(edges):
        _pass_msg(fgraph, e, e.target(), e.source(), which)

# pass messages root-leaves
    for e in edges:
        _pass_msg(fgraph, e, e.source(), e.target(), which)

    _, fgraph.logZ = fgraph.pr(fgraph.variables[0], with_log_norm=True)
    logger.debug('logZ: %s', fgraph.logZ)

def _pass_msg(fgraph, e, s, t, which):
    snode = fgraph.vp_node[s]
    tnode = fgraph.vp_node[t]
    logger.debug('passing %s -> %s', snode, tnode)
    logger.debug('passing %s -> %s', s, t)

    s_vtype = fgraph.vp_type[s]
    if s_vtype == 'variable':
        arity = fgraph.vp_arity[s]
        sp_msg, sp_msg_lnc = np.ones(arity), 0
        mp_msg, mp_msg_lnc = np.ones(arity), 0

        logger.debug(' * sp_msg:    %s', sp_msg)
        logger.debug(' * sp_msg_lnc: %s', sp_msg_lnc)
        logger.debug(' * mp_msg:    %s', mp_msg)
        logger.debug(' * mp_msg_lnc: %s', mp_msg_lnc)
        neighbours = list(s.out_neighbours())
        logger.debug(' * neighbouts before: %s', neighbours)
        neighbours.remove(t)
        logger.debug(' * neighbouts before: %s', neighbours)
        for neigh in neighbours:
            neigh_edge = fgraph.graph.edge(neigh, s)
            logger.debug('   -------')
            if 'sum-product' in which:
                # print fgraph.ep_sp_msg_fv[neigh_edge]
                sp_msg    *= fgraph.ep_sp_msg_fv[neigh_edge]
                sp_msg_lnc += fgraph.ep_sp_msg_fv_lnc[neigh_edge]
                logger.debug(' * sp_msg:    %s', sp_msg)
                logger.debug(' * sp_msg_lnc: %s', sp_msg_lnc)
            if 'max-product' in which:
                mp_msg    *= fgraph.ep_mp_msg_fv[neigh_edge]
                mp_msg_lnc += fgraph.ep_mp_msg_fv_lnc[neigh_edge]
                logger.debug(' * mp_msg:    %s', mp_msg)
                logger.debug(' * mp_msg_lnc: %s', mp_msg_lnc)

        if 'sum-product' in which:
            sp_msg_lnc += math.log(sp_msg.sum())
            fgraph.ep_sp_msg_vf[e] = sp_msg/sp_msg.sum()
            fgraph.ep_sp_msg_vf_lnc[e] = sp_msg_lnc
        if 'max-product' in which:
            mp_msg_lnc += math.log(mp_msg.sum())
            fgraph.ep_mp_msg_vf[e] = mp_msg/mp_msg.sum()
            fgraph.ep_mp_msg_vf_lnc[e] = mp_msg_lnc
    elif s_vtype  == 'factor':
        nname = fgraph.vp_name[t]
        msgs, msgs_lnc = {}, {}
        if 'sum-product' in which:
            msgs['sum-product'] = { nname: np.ones(fgraph.vp_arity[t]) }
            msgs_lnc['sum-product'] = 0
        if 'max-product' in which:
            msgs['max-product'] = { nname: np.ones(fgraph.vp_arity[t]) }
            msgs_lnc['max-product'] = 0

        # logger.debug(' * in_msg:    %s', msgs[nname])
        # logger.debug(' * in_msg_lnc: %s', msg_lnc)

        neighbours = list(s.out_neighbours())
        # neighbours.remove(t)
        for neigh in filter(lambda neigh: neigh != t, neighbours):
            nname = fgraph.vp_name[neigh]
            neigh_edge = fgraph.graph.edge(neigh, s)
            if 'sum-product' in which:
                msgs['sum-product'][nname] = fgraph.ep_sp_msg_vf[neigh_edge]
                msgs_lnc['sum-product'] += fgraph.ep_sp_msg_vf_lnc[neigh_edge]
            if 'max-product' in which:
                msgs['max-product'][nname] = fgraph.ep_mp_msg_vf[neigh_edge]
                msgs_lnc['max-product'] += fgraph.ep_mp_msg_vf_lnc[neigh_edge]

            # logger.debug(' * in_msg:    %s', fgraph.ep_mp_msg_vf[neigh_edge])
            # logger.debug(' * in_msg_lnc: %s', fgraph.ep_mp_msg_vf_lnc[neigh_edge])

        if 'sum-product' in which:
            msgs['sum-product'] = [ msgs['sum-product'][n] for n in fgraph.vp_table_inputs[s] ]
        if 'max-product' in which:
            msgs['max-product'] = [ msgs['max-product'][n] for n in fgraph.vp_table_inputs[s] ]

        logger.debug(' * factor: %s', fgraph.vp_table[s])
        logger.debug(' * msgs: %s', msgs)
        # logger.debug(' * reduce(*): %s', reduce(np.multiply, np.ix_(*msgs)))

        prod = {}
        if 'sum-product' in which:
            prod['sum-product'] = fgraph.vp_table[s] * reduce(np.multiply, np.ix_(*msgs['sum-product']))
        if 'max-product' in which:
            prod['max-product'] = fgraph.vp_table[s] * reduce(np.multiply, np.ix_(*msgs['max-product']))

        axis = fgraph.vp_table_inputs[s].index(fgraph.vp_name[t])
        negaxis = tuple(filter(lambda a: a!= axis, range(len(neighbours))))

        if 'sum-product' in which:
            sp_msg = prod['sum-product'].sum(axis = negaxis)
            sp_msg_sum = sp_msg.sum()
            fgraph.ep_sp_msg_fv_lnc[e] = msgs_lnc['sum-product'] + math.log(sp_msg_sum)
            fgraph.ep_sp_msg_fv[e] = sp_msg/sp_msg_sum
            logger.debug(' * out_msg_sp:    %s', fgraph.ep_sp_msg_fv[e])
            logger.debug(' * out_msg_sp_lnc: %s', fgraph.ep_sp_msg_fv_lnc[e])
        if 'max-product' in which:
            mp_msg = prod['max-product'].max(axis = negaxis)
            mp_msg_sum = mp_msg.sum()
            fgraph.ep_mp_msg_fv_lnc[e] = msgs_lnc['max-product'] + math.log(mp_msg_sum)
            fgraph.ep_mp_msg_fv[e] = mp_msg/mp_msg_sum
            logger.debug(' * out_msg_mp:    %s', fgraph.ep_mp_msg_fv[e])
            logger.debug(' * out_msg_mp_lnc: %s', fgraph.ep_mp_msg_fv_lnc[e])
    else:
        raise Exception('variable type error: {}'.format(s_vtype))

