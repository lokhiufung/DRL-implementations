import agents
from env_helper import Transition
from utils import get_logger


class Trainer(object):
    def __init__(
        self, agent_name, env_name, agent_config, 
        logger=None, fh_lv='error', ch_lv='debug'
        ):
        self.agent = getattr(agents, agent_name)(**agent_config)
        self.env = gym.make(env_name)
        self._transition = None
        self._num_traceback = agent_config.get('n_steps', 1) 
        
        if logger is None:
            self.logger = get_logger('trainer', fh_lv=fh_lv, ch_lv=ch_lv)

    def run_train_episodes(self, num_episodes, render=False):
        if self._transition is None:
            self._transition = Transition(num_traceback=self._num_traceback)
        for episode in range(num_episodes):
            while True:
                if render:
                    env.render()
                action = agent.epsilon_greedy_infer(state, return_tensor=False)
                next_state, reward, done, info = env.step(action)
                transition.append(state, reward, action, next_state)
                
                # save to memeory
                if transition.is_ready:
                    # get training samples from transotion object
                    state, action, summed_reward, next_state = transition.get_aggregated()
                    agent.remember_to_replay_buffer(state, action, summed_reward)
                    agent.remember_to_dnd(state, action, summed_reward)        

                # learning
                agent.replay(batch_size)
                if done:    
                    break

    def run_warmup_episodes(self, num_episodes, render=False):
        if self._transition is None:
            self._transition = Transition(num_traceback=self._num_traceback)
        for episode in range(num_episodes):
            while True:
                if render:
                    env.render()
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                transition.append(state, reward, action, next_state)
                
                # save to memeory
                if transition.is_ready:
                    # get training samples from transotion object
                    state, action, summed_reward, next_state = transition.get_aggregated()
                    agent.remember_to_replay_buffer(state, action, summed_reward)
                    agent.remember_to_dnd(encoded_state, action, summed_reward)        

                if done:    
                    break
        

