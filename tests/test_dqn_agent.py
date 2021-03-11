from omegaconf import DictConfig
import numpy as np
import torch

from drl.agents.dqn_agent import DQNAgent

def get_dqn_agent():
    cfg = {
        'warmup': 1000,
        'train_data': {
            'dataset': {
                'batch_size': 4
            }
        },
        'env': {
            'env_name': 'CartPole-v0',
        },
        'exploration_scheduler': {
            '_target_': 'drl.core.exploration.EpsilonGreeyExplorationScheduler',
            'eps_start': 0.9,
            'eps_end': 0.1,
            'decay_factor': 1.0
        },
        'replay_buffer': {
            'capacity': 1e6
        },
        'network': {
            'encoder': {
                '_target_': 'drl.blocks.encoder.dense_encoder.DenseEncoder',
                'n_layers': 2,
                'input_dim': 4,
                'hidden_dim': 64,
            },
            'output_head': {
                '_target_': 'drl.blocks.heads.ValueHead',
                'output_dim': 2,
                'hidden_dim': 64    
            }
        },
        'optimizer': {
            '_target_': 'torch.optim.RMSprop',
            'lr': 1e-3,
        }
    }
    agent = DQNAgent(cfg=DictConfig(cfg))
    return agent


def test_dqn_agent_play_step():
    agent = get_dqn_agent()
    agent.play_step()

    assert agent.agent_steps == 1
    assert len(agent.replay_buffer) == 1


def test_dqn_agent_act():
    agent = get_dqn_agent()
    
    state = agent._env.current_state
    with torch.no_grad():
        value, action = agent.act(state)
    assert type(action) == np.int64

