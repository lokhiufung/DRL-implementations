from abc import ABC, abstractmethod, abstractproperty


class RewardScaler(ABC):
    def __init__(self):
        self._n = 1  # start with 1, update after estimate updates

    def scale(self, reward):
        self._update_estimate(reward)
        return self._scale(reward)

    @abstractmethod
    def _scale(self, reward):
        """how to scale reward after updating the estimates"""
        pass

    def _update_estimate(self, reward):
        """how to update the estimates needed to scale the reward"""
        pass

    @abstractproperty
    def estimates(self) -> dict:
        """return the estimates"""
        pass

    @property
    def n_updated_steps(self):
        return self._n