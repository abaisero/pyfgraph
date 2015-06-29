#! /usr/bin/python

import unittest

import pyfgraph.samples.tabgraph as tabgraph
import pyfgraph.samples.logger as logger
import pyfgraph.algo as algo

class TabGraphTestCase(unittest.TestCase):
    def setUp(self):
        logger.setup_file_logger('log.test_tabgraph.log')
        self.fg = tabgraph.simple_tabgraph()

    def test_viterbi(self):
        e = 1e-5
        algo.message_passing(self.fg, 'sum-product', 'max-product')
        self.assertTrue(1000. - e < self.fg.max() < 1000. + e)
        self.assertEqual(self.fg.argmax(), [0, 0, 1])

if __name__ == '__main__':
    unittest.main()
