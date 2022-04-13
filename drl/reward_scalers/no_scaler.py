from drl.core.reward_scaler import RewardScaler


class NoScaler(RewardScaler):
    def _scale(self, reward):
        return reward

    @property
    def estimates(self) -> dict:
        return {}

            