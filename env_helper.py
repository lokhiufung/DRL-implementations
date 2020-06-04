"""
Objects that store part of the transition history for further manipulation
(e.g multisteps value estimate)
"""
from collections import deque


class Transition(object):
    def __init__(self, num_traceback):
        self.num_traceback = num_traceback
        self._history = deque(maxlen=self.num_traceback)
        self._is_ready = False

    @property
    def is_ready(self):
        if not self._is_ready:
            if len(self._history) == self.num_traceback:
                self._is_ready = True
        return self._is_ready

    def __len__(self):
        return len(self._history)

    def append(self, state, reward, action, next_state):
        self._history.append((state, reward, action, next_state))
        
    
    def clean_history(self):
        self._history.clear()
        
    def get_aggregated(self):
        start_state = self._history[0][0]
        start_action = self._history[0][2]
        summed_reward = sum([self._history[i][1] for i in range(len(self._history))])
        final_state = self._history[-1][3]
        return start_state, start_action, summed_reward, final_state

    def get_raw(self):
        pass        