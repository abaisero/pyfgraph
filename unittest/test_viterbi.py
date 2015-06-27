#! /usr/bin/python

import unittest

import pyfgraph.samples.tabgraph as tabgraph
import pyfgraph.algo as algo

class ViterbiTestCase(unittest.TestCase):
    def setUp(self):
        self.fg = tabgraph.simple_tabgraph()
        algo.message_passing(self.fg, 'sum-product', 'max-product')

    def test_viterbi(self):
        e = 1e-5
        self.assertTrue(1000. - e < self.fg.max() < 1000. + e)
        self.assertEqual(self.fg.argmax(), [0, 0, 1])

if __name__ == '__main__':
    unittest.main()
