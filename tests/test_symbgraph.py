#! /usr/bin/python

import unittest

import pyfgraph.samples.symbgraph as symbgraph
import pyfgraph.utils.log as log

class SymbGraphTestCase(unittest.TestCase):
    def setUp(self):
        log.setup_file_logger('log.test_symbgraph.log')
        self.fg, self.make_data = symbgraph.simple_symbgraph()

    def test_train(self):
        data = self.make_data(n=10)
        self.fg.train(data)

        vit = self.fg.viterbi(data)
        for x, y, v in zip(data['X'], data['Y'], vit):
            print x, y, v

if __name__ == '__main__':
    unittest.main()

