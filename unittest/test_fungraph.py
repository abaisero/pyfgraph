#! /usr/bin/python

import unittest

import pyfgraph.samples.fungraph as fungraph
import pyfgraph.samples.logger as logger

class FunGraphTestCase(unittest.TestCase):
    def setUp(self):
        logger.setup_file_logger('log.test_fungraph.log')
        self.fg, self.make_data = fungraph.simple_fungraph()

    def test_train(self):
        data = self.make_data(n=100)
        self.fg.train(data)

if __name__ == '__main__':
    unittest.main()
