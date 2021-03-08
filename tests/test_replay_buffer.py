import gym

from drl.blocks.memory.replay_buffer import ReplayBuffer


ENV = gym.make('CartPole-v0')


def test_replay_buffer_append():
    replay_buffer = ReplayBuffer(capacity=1e6)
    
    # 1 step
    state = ENV.reset()

    action = ENV.action_space.sample()
    next_state, reward, done, _ = ENV.step(action)

    replay_buffer.append(state, reward, next_state, action, done)
    assert len(replay_buffer) == 1



        