class Feats(object):
    nfeats = 0

    def __init__(self, name, nfeats=None):
        self.name = name
        self.nfeats = nfeats
        self.fslice = slice(None)
        if nfeats is not None:
            self.fslice = slice(Feats.nfeats, Feats.nfeats+nfeats)
            Feats.nfeats += self.nfeats

    def make(self, feats):
        if self.nfeats is None:
            self.nfeats = Feats.nfeats
        self.feats = feats[self.fslice]

    @classmethod
    def extend(self, data):
        pass

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

