import gym


class GymEnvironment(Environment):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def reset(self):
        return self.env.reset()
        
    def render(self):
        return

    def step(self, action):
        return self.env.step()

    def close(self):
        return self.env.close()