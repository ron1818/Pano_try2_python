
class Options(dict):
    """Make Dict to property like options to replace argparse

    Args:
        dict (string:obj): option dictionary
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def add_argument(self, name, value, **kwargs):
        self[name] = value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)