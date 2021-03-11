import gym

from drl.core.enviroment import Environment


env_config = {
    'env_name': 'CartPole-v0',
}
openai_env = gym.make(env_config['env_name'])
env = Environment(openai_env)


def test_env_cum_reward_in_episode():
    cum_reward = 0.0
    env.reset()
    while True:
        _, reward, done, _ = env.step(env.sample_action())
        cum_reward += reward
        if done:
            break
    assert env.cum_reward_in_episode == cum_reward


def test_env_reward_per_episode():
    env.reset()
    reward_per_episode = []
    for _ in range(10):
        cum_reward = 0.0
        while True:
            _, reward, done, _ = env.step(env.sample_action())
            cum_reward += reward
            if done:
                break
        env.reset_for_next_episode()
        reward_per_episode.append(cum_reward)
    
    assert len(env.reward_per_episode) == len(reward_per_episode)
    print(env.reward_per_episode)
    print(reward_per_episode)
    for r_env, r in zip(env.reward_per_episode, reward_per_episode):
        assert r_env - r < 1e-5

# test_env_reward_per_episode()