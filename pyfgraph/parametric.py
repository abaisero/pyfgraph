class Feats(object):
    nfeats = 0

    def __init__(self, name, nfeats):
        self.name = name
        self.nfeats = nfeats

    def make(self, feats):
        pass

    @classmethod
    def clean(cls):
        cls.nfeats = 0

class Params(object):
    nparams = 0

    def __init__(self, name, nparams=None):
        self.name = name
        self.nparams = nparams
        self.pslice = slice(None)
        if nparams is not None:
            self.pslice = slice(Params.nparams, Params.nparams+nparams)
            Params.nparams += self.nparams

    def make(self, params):
        if self.nparams is None:
            self.nparams = Params.nparams
        self.params = params[self.pslice]

    @classmethod
    def clean(cls):
        cls.nparams = 0

