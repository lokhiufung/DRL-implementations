from typing import Tuple

import numpy as np
import gym

class Environment(object):
    """An wrapper object of openai-gym like environment for agent trainer"""
    def __init__(self, env):
        self._env = env

        self.n_episodes = 1
        # initialize game statistics
        self.reward_per_episode = []
        # self.cum_reward_across_episode = 0.0
        self.cum_reward_in_episode = 0.0

        self._current_state = self._env.reset()  # initialize reset()

    @property
    def current_state(self) -> np.array:
        return self._current_state

    def reset_for_next_episode(self) -> np.array:
        cum_reward = self.cum_reward_in_episode
        self.reward_per_episode.append(cum_reward)
        self.cum_reward_in_episode = 0.0
        self.n_episodes += 1
        return self._env.reset()

    def reset(self) -> np.array:
        self.n_episodes = 1
        self.cum_reward_in_episode = 0.0
        self.reward_per_episode = []
        return self._env.reset()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def step(self, action: int) -> Tuple[np.array, float, bool, str]:
        state, reward, done, info = self._env.step(action)
        self._current_state = state
        self.cum_reward_in_episode += reward
        return state, reward, done, info

    def sample_action(self) -> int:
        return self._env.action_space.sample()

    def close(self):
        return self._env.close()
    

