from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl

from drl.agents.dqn_agent import DQNAgent


@hydra.main(config_name='configs/dqn_agent.yaml')
def main(cfg: DictConfig):
    agent = DQNAgent(cfg.agent)

    agent.warmup(100)  # warmup with 100 episodes

    print('len: {}'.format(len(agent.replay_buffer)))
    trainer = pl.Trainer(**cfg.trainer)

    trainer.fit(agent)


if __name__ == '__main__':
    main()
    