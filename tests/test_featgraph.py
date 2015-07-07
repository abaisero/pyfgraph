#! /usr/bin/python

import unittest

import numpy as np

import pyfgraph.samples.featgraph as featgraph
import pyfgraph.samples.logger as logger
import pyfgraph.algo as algo

class FeatGraphTestCase(unittest.TestCase):
    def setUp(self):
        logger.setup_file_logger('log.test_featgraph.log')

    # def test_simple_message_passing(self):
    #     e = 1e-5
    #     fg, make_data = featgraph.simple_featgraph()
    #     data = make_data(n=10)
    #     fg.train(data)

    #     v_msgs = algo.check_message_passing(fg)
    #     for v, msgs in zip(fg.variables, v_msgs):
    #         mdiff = (msgs.max(axis=0) - msgs.min(axis=0)).max()
    #         self.assertTrue(mdiff<e)

# TODO for some reason, the second graphs inherits the "feats" params of the previous one
    # def test_domain_message_passing(self):
    #     e = 1e-5
    #     fg, make_data = featgraph.domain_featgraph()
    #     data = make_data(n=10)
    #     fg.train(data)

    #     v_msgs = algo.check_message_passing(fg)
    #     for v, msgs in zip(fg.variables, v_msgs):
    #         mdiff = (msgs.max(axis=0) - msgs.min(axis=0)).max()
    #         self.assertTrue(mdiff<e)

    def test_gradient_check(self):
        e = 1e-5
        fg, make_data = featgraph.simple_featgraph()
        data = make_data(n=10)
        fg.check_gradient = True
        err_grad = fg.train(data)

        self.assertTrue(err_grad.max()<e)
        
if __name__ == '__main__':
    unittest.main()
