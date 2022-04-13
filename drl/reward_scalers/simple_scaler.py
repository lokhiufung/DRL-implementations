from drl.core.reward_scaler import RewardScaler


class SimpleScaler(RewardScaler):
    def __init__(self, highest_score):
        super().__init__()

        self.highest_score = highest_score

    def _scale(self, reward):
        reward = reward / self.highest_score
        return reward

    @property
    def estimates(self):
        return {
            'highest_score': self.highest_score
        }
    