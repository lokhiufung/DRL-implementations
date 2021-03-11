
from omegaconf import DictConfig 
import pytorch_lightning as pl
from hydra.utils import instantiate
import gym

from drl.core.enviroment import Environment


class Agent(pl.LightningModule):
    """
    Agent is constituted of Modules and interact with Environment 
    """
    
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
    
        self.setup_environment(cfg.env)

        # self.warmup(n_episodes=cfg.warmup)

    def setup_exploration_scheduler(self, exploration_cfg):
        # self._exploration_scheduler = instantiate(exploration_cfg, agent_steps=self.agent_steps)
        self._exploration_scheduler = instantiate(exploration_cfg)

        
    def setup_optimizers(self, optim_cfg):
        raise NotImplementedError

    def setup_environment(self, env_cfg):
        openai_env = gym.make(env_cfg.env_name)
        self._env = Environment(env=openai_env)

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

    def play_step(self):
        raise NotImplementedError
            
    def warmup(self, n_episodes: int):
        """get samples by interacting with the environment"""
        while self._env.n_episodes > n_episodes:
            self.play_step()
        




        
    