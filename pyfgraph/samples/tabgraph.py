from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, TabFactor
from pyfgraph.algo import message_passing
import pyfgraph.utils.log as log

def simple_tabgraph():
    fg = FactorGraph()

    V1 = fg.add(Variable, 'V1', arity=2)
    V2 = fg.add(Variable, 'V2', arity=2)
    V3 = fg.add(Variable, 'V3', arity=2)

    F1 = fg.add(TabFactor, 'F1', V1          )
    F2 = fg.add(TabFactor, 'F2', (V1, V2)    )
    F3 = fg.add(TabFactor, 'F3', (V2, V3)    )

# F1 prefers if V1 is 0
    F1.table = [ 10, 1 ]

# F2 prefers if V1 and V2 are the same
    F2.table = [[ 10, 1 ],
                [ 1, 10 ]]

# F3 prefers if V2 and V3 are different
    F3.table = [[ 1, 10 ],
                [ 10, 1 ]]

    fg.make()
    return fg

if __name__ == '__main__':
    log.setup_file_logger('log.tabgraph.log')
    fg = simple_tabgraph()
    message_passing(fg, 'max-product', 'sum-product')

    print 'max:    {}'.format(fg.max())
    print 'argmax: {}'.format(fg.argmax())

