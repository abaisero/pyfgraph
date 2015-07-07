#! /usr/bin/python

import unittest

import pyfgraph.samples.tabgraph as tabgraph
import pyfgraph.samples.logger as logger
import pyfgraph.algo as algo

class TabGraphTestCase(unittest.TestCase):
    def setUp(self):
        logger.setup_file_logger('log.test_tabgraph.log')

    def test_simple_viterbi(self):
        e = 1e-5
        fg = tabgraph.simple_tabgraph()
        algo.message_passing(fg, 'sum-product', 'max-product')
        self.assertTrue(1000. - e < fg.max() < 1000. + e)
        self.assertEqual(fg.argmax(), [0, 0, 1])

    def test_domain_viterbi(self):
        e = 1e-5
        fg = tabgraph.domain_tabgraph()
        algo.message_passing(fg, 'sum-product', 'max-product')
        self.assertTrue(1000. - e < fg.max() < 1000. + e)
        self.assertEqual(fg.argmax(), ['This', 'Code', 'Rules'])

if __name__ == '__main__':
    unittest.main()
