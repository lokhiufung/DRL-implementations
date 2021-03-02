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
        
    def act(self):
        raise NotImplementedError




        
    