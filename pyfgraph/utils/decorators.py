import inspect

def kwargsdec(f):
    """ Allows to call functions over-specifying the number of key-word arguments """
    def wrapper(**kwargs):
        args = inspect.getargspec(f).args
        return f(**{ k: kwargs[k] for k in args})
    return wrapper

