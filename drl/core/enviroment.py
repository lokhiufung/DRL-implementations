import gym

class Environment(object):
    """"""
    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    

