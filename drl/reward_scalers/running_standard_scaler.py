
from drl.core.reward_scaler import RewardScaler



class RunningStandardScaler(RewardScaler):
    def __init__(self):
        super().__init__()
        
        self.mean = 0.0
        self.var = 0.0

    def _scale(self, reward):
        reward = (reward - self.mean) / self.var**0.5
        return reward

    def _update_estimate(self, reward):
        self.mean += (reward - self.mean) / self._n
        self.var += ((reward - self.mean)**2 - self.var) / self._n

        self._n += 1

    @property
    def estimates(self):
        return {
            'mean': self.mean,
            'var': self.var,
        }
