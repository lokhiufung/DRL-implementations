import gym
from gym import envs


def get_openai_gym_env_ids():
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    return env_ids


def check_is_openai_gym(env_id):
    env_ids = get_openai_gym_env_ids()
    if env_id in env_ids:
        return True
    else:
        return False


class Environment:
    def __init__(self, env_id, use_pixel=False):
        self.env_id = env_id
        self.is_openai_gym = check_is_openai_gym(env_id)
        if self.is_openai_gym:
            self.env = gym.make(self.env_id)
        self.use_pixel = use_pixel

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
    
    def _get_pixel_state(self):
        pass        
    
    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
    