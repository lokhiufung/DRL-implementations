
class ExplorationScheduler:
    """scheduler helper for exploration parameters"""

class EpsilonGreeyExplorationScheduler(ExplorationScheduler):
    def __init__(self, agent_steps, eps_start, eps_end, decay_factor):
        self._eps = eps_start
        self._eps_start = eps_start
        self._eps_end = eps_end

    @property
    def eps(self):
        return self._eps

    