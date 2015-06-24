import numpy as np

import pyfgraph
from pyfgraph.params import Params
from pyfgraph.nodes import Node, Variable, Factor, FFactor
from pyfgraph.fgraph import FactorGraph

if __name__ == '__main__':
    fg = FactorGraph()

    V1 = fg.add(Variable, 'var_1', arity=2)
    V2 = fg.add(Variable, 'var_2', arity=2)
    V3 = fg.add(Variable, 'var_3', arity=2)

    F1 = fg.add(Factor, 'f_1', V1          )
    F2 = fg.add(Factor, 'f_2', (V1, V2)    )
    F3 = fg.add(Factor, 'f_3', (V2, V3)    )
# TODO This factor requires loopy bp
    # F4 = fg.add(Factor, 'f_4', (V1, V2, V3))

# F1 prefers if V1 is 0
    F1.table = np.array([ 10, 1 ])

# F2 prefers if V1 and V2 are the same
    F2.table = np.array([[ 10, 1 ],
                         [ 1, 10 ]])

# F3 prefers if V2 and V3 are different
    F3.table = np.array([[ 1, 10 ],
                         [ 10, 1 ]])

# # F4 prefers if exactly one of the variables is 1
#     F4.table = np.array([[[ 1, 10 ],
#                           [ 10, 1 ]],
#                          [[ 10, 1 ],
#                           [ 1,  1 ]]])

# uncomment the following to change F4, and watch out for the different output
# # F4 prefers if exactly two of the variables are 1
#     F4.table = np.array([[[ 1, 1 ],
#                           [ 1, 10 ]],
#                          [[ 1, 10 ],
#                           [ 10, 1 ]]])

    fg.make(done=True)

    fg.message_passing()

    print ' === Message Checking === '
    fg.check_message_passing()

    print ' === Viterbi === '
    print 'argmax: {}'.format(fg.argmax())
    print 'max: {}'.format(fg.max())

    print ' === Partition Function === '
    print 'Z: {}'.format(fg.Z)
    print 'logZ: {}'.format(fg.logZ)
