from drl.core.reward_scaler import RewardScaler


class MinMaxScaler(RewardScaler):
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high
        self._n = 1  # start with 1, update after estimate updates

    def _scale(self, reward):
        reward = (reward - self.low) / (self.high - self.low)
        return reward
        
    @property
    def estimates(self):
        return {
            'high': self.high,
            'low': self.low,
        }
