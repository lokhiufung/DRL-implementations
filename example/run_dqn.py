from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from drl.agents.dqn_agent import DQNAgent


@hydra.main(config_name='configs/dqn_agent.yaml')
def main(cfg: DictConfig):
    agent = DQNAgent(cfg.agent)
    model_ckpt_callback = ModelCheckpoint(monitor='train_loss')

    agent.warmup(1000)  # warmup with 1000 episodes

    trainer = pl.Trainer(**cfg.trainer, callbacks=[model_ckpt_callback])

    trainer.fit(agent)


if __name__ == '__main__':
    main()
    