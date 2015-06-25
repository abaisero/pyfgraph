import itertools as itt

def proditer(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()

    for vtuple in itt.product(*values):
        yield { k: v for k, v in zip(keys, vtuple) }

