from pyfgraph.fgraph import FactorGraph
from pyfgraph.nodes import Variable, TabFactor
from pyfgraph.algo import message_passing
import logging

def simple_tabgraph():
    fg = FactorGraph()

    # V1 = fg.add(Variable, 'V1', domain=2)
    # V2 = fg.add(Variable, 'V2', domain=2)
    # V3 = fg.add(Variable, 'V3', domain=2)
    V1 = fg.add(Variable, 'V1', domain=['This', 'Shitty'])
    V2 = fg.add(Variable, 'V2', domain=['Code', 'Breaks'])
    V3 = fg.add(Variable, 'V3', domain=['All The', 'Rules'])

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
    fmt = '%(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    fmt = '%(asctime)s %(levelname)s @%(lineno)d:%(filename)s - %(funcName)s(): %(message)s'
    logging.basicConfig(filename='log.featgraph.log',
                        filemode='w',
                        format=fmt,
                        level=logging.DEBUG)

    fg = simple_tabgraph()
    message_passing(fg, 'max-product', 'sum-product')

    print 'max:    {}'.format(fg.max())
    print 'argmax: {}'.format(fg.argmax())

