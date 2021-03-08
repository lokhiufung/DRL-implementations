from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl

from drl.agents.dqn_agent import DQNAgent


def main(cfg: DictConfig):
    agent = DQNAgent(cfg)
    trainer = pl.Trainer(agent)

    trainer.fit(agent)


if __name__ == '__main__':
    main()
    