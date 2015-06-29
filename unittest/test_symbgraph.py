#! /usr/bin/python

import unittest

import pyfgraph.samples.symbgraph as symbgraph
import pyfgraph.samples.logger as logger

class SymbGraphTestCase(unittest.TestCase):
    def setUp(self):
        logger.setup_file_logger('log.test_symbgraph.log')
        self.fg, self.make_data = symbgraph.simple_symbgraph()

    def test_train(self):
        data = self.make_data(n=100)
        self.fg.train(data)

if __name__ == '__main__':
    unittest.main()
