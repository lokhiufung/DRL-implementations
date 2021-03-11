import math


class ExplorationScheduler:
    """scheduler helper for exploration parameters"""


class EpsilonGreeyExplorationScheduler(ExplorationScheduler):
    def __init__(self, eps_start, eps_end, decay_factor):
        # self._steps = agent_steps
        # self._eps = eps_start
        self._eps_start = eps_start
        self._eps_end = eps_end

        self.decay_factor = decay_factor

    # def _get_eps(self):
    #     return max(
    #         self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1 * self._steps / self.decay_factor),
    #         self._eps_end
    #     )

    # @property
    # def steps(self):
    #     return self._steps

    # @property
    # def eps(self):
    #     return self._get_eps()
    
    def get_eps_on_step(self, agent_steps: int):
        """temp: need to pass agent_steps every time"""
        return max(
            self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1 * agent_steps / self.decay_factor),
            self._eps_end
        )


    