import logging
logger = logging.getLogger(__name__)
print __name__

# NB: this does not in itself contain the parameters. It just represents the shared parameters
# Q: where is the actual parameter array? in the fg.
class Params(object):
    tot_nparams = 0
    def __init__(self, n):
        self.n = n
        self.pslice = slice(Params.tot_nparams, Params.tot_nparams+n)
        Params.tot_nparams += n


