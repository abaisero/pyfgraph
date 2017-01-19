import unittest
from fabric.api import *

def test():
    tests = unittest.TestLoader().discover('unittest')
    unittest.TextTestRunner().run(tests)
