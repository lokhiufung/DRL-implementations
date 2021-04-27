import gym

from drl.core.enviroment import Environment
from drl.core.utils import name_import


class GymEnvironment(gym.Env, Environment):
    """"""
    def __init__(self, env_name):
        self.env_name = env_name
        self._env = gym.make(self.env_name)
        if apply_wrapper is not None:
            if isinstance(apply_wrapper, str):
                wrapper_module = name_import(f'drl.environments.gym_wrappers')
                wrapper_class = getattr(wrapper_module, apply_wrapper)
            # elif isinstance(apply_wrapper, gym.Wrapper):
            wrapper_class = apply_wrapper
            # else:
            #     raise ValueError('apply_wrapper should neither be a gym.Wrapper object or Wrapper name.')
            self._env = wrapper_class(self._env)

    def reset(self):
        return self._env.reset()
        
    def render(self, mode='human'):
        return self._env.render(mode)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

