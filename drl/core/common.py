import collections

from drl.core.utils import load_json, load_yaml


class Serializable(object):
    """

    """



class HyperParameters(collections.UserDict):
    def __init__(self, params):
        super().__init__(params)

    def from_json(cls, filepath):
        params = load_json(filepath)
        return cls(params)

    def from_yaml(cls, filepath):
        params = load_yaml(filepath):
        return cls(params)


    # def __add__(self, hyperparams):
    #     added_params = {self.name: self, hyperparams.name: hyperparams}
    #     return HyperParameters(added_params)




