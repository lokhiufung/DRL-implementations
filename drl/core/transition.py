from typing import Tuple
from collections import namedtuple, deque

import numpy as np


Transition = namedtuple(
    'Transition', [
        'state',
        'action',
        'reward',
        'next_state',
        'done',
    ]
)

class TransitionHistory:
    def __init__(self, n_transitions):
        self.n_transitions = n_transitions
        self.transitions = deque(maxlen=self.n_transitions)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return list(self.transitions)[idx]
        
    def get_rewards(self):
        rewards = [transition.reward for transition in self.transitions]
        return rewards

    def get_transitions(self):
        return self.transitions
    
    def get_state_action_rewards(self) -> Tuple[np.ndarray, list]:
        """return the state and the rewards collected since the state
        """
        transitions = list(self.transitions)
        state = transitions[0].state
        action = transitions[0].action
        rewards = self.get_rewards()
        return state, action, rewards

    def check_done(self):
        transitions = list(self.transitions)
        done = transitions[-1].done
        return done

    def append(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.transitions.append(transition)
    
    def reset(self):
        # reset all the transitions (i.e at the beginning of each episode)
        self.transitions = deque(maxlen=self.n_transitions)
