import logging

fmt = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_file_logger(fname = 'log.log', level = logging.DEBUG):
    logging.basicConfig(filename=fname,
                        filemode='w',
                        format=fmt,
                        level=level)

def setup_stream_logger(level = logging.INFO):
    logger = logging.getLogger()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def log_values(fgraph):
    pass
    # logger = logging.getLogger()
    # for v in fgraph.variables:
    #     vnode = fgraph.vp_node[v]
    #     logger.info('%s.value: %s', vnode.name, str(vnode.value))

def log_params(fgraph):
    pass
    # logger = logging.getLogger()
    # logger.info('Params: %s', str(fgraph.params))

def log_tables(fgraph):
    pass
    # logger = logging.getLogger()
    # logger.debug('Current Tables:')
    # for f in fgraph.factors:
    #     fnode = fgraph.vp_node[f]
    #     logger.debug('%s.table: %s', fnode.name, str(fnode.table))

def log_messages(fgraph, which):
    pass
    # logger = logging.getLogger()
    # logger.info('Number of Edges: %s', len([ e for e in fgraph.graph.edges()]))
    # logger.info(' Current Messages:')
    # for w in which:
    #     for e in fgraph.graph.edges():
    #         s, t = e.source(), e.target()
    #         snode = fgraph.vp_node[s]
    #         tnode = fgraph.vp_node[t]

    #         if fgraph.vp_type[s] == 'variable':
    #             if w == 'sum-product':
    #                 msg = fgraph.ep_sp_msg_vf[e]
    #             if w == 'max-product':
    #                 msg = fgraph.ep_mp_msg_vf[e]
    #         else:
    #             if w == 'sum-product':
    #                 msg = fgraph.ep_sp_msg_fv[e]
    #             if w == 'max-product':
    #                 msg = fgraph.ep_mp_msg_fv[e]

    #         logger.info(' %s.msg %s -> %s: %s', w, snode, tnode, str(msg))

    #         if fgraph.vp_type[s] == 'variable':
    #             if w == 'sum-product':
    #                 msg = fgraph.ep_sp_msg_fv[e]
    #             if w == 'max-product':
    #                 msg = fgraph.ep_mp_msg_fv[e]
    #         else:
    #             if w == 'sum-product':
    #                 msg = fgraph.ep_sp_msg_vf[e]
    #             if w == 'max-product':
    #                 msg = fgraph.ep_mp_msg_vf[e]

    #         logger.info(' %s.msg %s -> %s: %s', w, tnode, snode, str(msg))

