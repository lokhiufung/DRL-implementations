
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
        self.haparams = cfg  # save cfg as hyparams

        self._train_dataset = None
        self._train_dataloader = None
        self._val_dataset = None
        self._val_dataloader = None
        self._optimizer = None

        self._env = None
        self._exploration_scheduler = None
        self.agent_steps = 0
        if cfg.exploration_scheduler:
            self.setup_exploration_scheduler(cfg.exploration_scheduler)

        self.mode = 'train'

    def setup_exploration_scheduler(self, exploration_cfg):
        self._exploration_scheduler = instantiate(exploration_cfg, agent_steps=self.agent_steps)
        
    def setup_optimizers(self, optim_cfg):
        raise NotImplementedError

    def setup_environment(self, env_cfg):
        raise NotImplementedError

    def setup_train_dataloader(self, train_cfg):
        raise NotImplementedError

    def setup_val_dataloader(self, val_cfg):
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
    
    def game_loop(self):
        pass

    def run_warmup_episodes(self, n_episodes: int):
        """get samples by interacting with the environment"""
        for episode in range(n_episodes):
            self.run_util_done()





        
    