from omegaconf import DictConfig 
import pytorch_lightning as pl


class Agent(pl.LightningModule):
    """
    Agent is constituted of Modules and interact with Environment 
    """
    # def serialize_modules(self, ckpt_dir):
    #     for module in self._modules:
    #         modules.serialize(ckpt_dir)
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._train_dataset = None
        self._train_dataloader = None
        self._val_dataset = None
        self._val_dataloader = None
        self._optimizer = None
    
    def setup_environment(self):
        raise NotImplementedError

    def setup_train_dataloader(self):
        raise NotImplementedError

    def setup_val_dataloader(self):
        raise NotImplementedError
    
    def train_dataloader(self):
        if self._train_dataloader:
            return self._train_dataloader
        else:
            raise AttributeError('please setup_train_dataloader() first.')

    def val_dataloader(self):
        if self._val_dataloader:
            return self._val_dataloader
        else:
            raise AttributeError('please setup_val_dataloader() first.')
    
    def act(self):
        raise NotImplementedError




        
    