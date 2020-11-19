from torch import nn


class Module:
    """
    Conceptual blocks of an agent's brain
    """
        

class NonTranableModule(Module):
    """
    Module with no trainable weights
    """
    _is_trainable = False

    @property
    def is_trainable(self):
        return self._is_trainable


class TrainableModule(Module):
    """
    Module with trainable weights
    """
    _is_trainable = True

    @property
    def is_trainable(self):
        return self._is_trainable

    @property
    def weights(self):
        return self._get_weights()
    
    def _get_weights(self):
        raise NotImplementedError

    @property
    def is_freeze(self):
        """
        whether 
        """
        raise NotImplementedError

    def freeze(self):
        """
        freeze all trainable weights
        """
        raise NotImplementedError


class TorchTrainableModule(TrainableModule, nn.Module):
    """
    This Module are parametrized and trained with pytorch
    """
    def forward(self):
        raise NotImplementedError

    def _get_weigths(self):
        return self.model_parameters()


