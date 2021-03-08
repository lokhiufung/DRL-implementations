from omegaconf import DictConfig

from drl.agents.dqn_agent import DQNAgent


def test_dqn_agent():
    cfg = {
        'warmup': 1000,
        'train_data': {
            'dataset': {
                'replay_buffer': 1e6,
                'batch_size': 4
            }
        },
        'env': {
            'env_name': 'CartPole-v0',
        },
        'exploration_scheduler': {
            '__target__': 'drl.core.exploration.EpsilonGreedyExplorationScheduler',
            'eps_start': 0.9,
            'eps_end': 0.1,
            'decay_factor': 1.0
        }
        'replay_buffer': {
            'capacity': 1e6
        },
        'network': {
            'encoder': {
                '__target__': 'drl.blocks.encoder.dense_encoder.DenseEncoder',
                'n_layers': 2,
                'input_dim': 4,
                'hidden_dim': 64,
                'output_dim': 64
            },
            'output_head': {
                '__target__': 'drl.blocks.heads.ValueHead',
                'output_dim': 2,
                'hidden_dim': 64    
            }
        },
        'optimizer': {
            '__target__': 'torch.optim.RMSprop',
            'lr': 1e-3,
        }
    }
    agent = DQNAgent(cfg=DictConfig(cfg))

