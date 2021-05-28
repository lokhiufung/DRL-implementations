import collections


__all__ = ['Transition', 'TransitionBuffer']


Transition = collections.namedtuple(
    'Transition',
    ['state', 'reward', 'next_state', 'action', 'done'],
)


class TransitionBuffer(collections.UserList):
    # @property
    # def transition(self):

    #     if len(self.buffer) >= self.n_steps:
    #         reward = 0.0
    #         for i, transition in enumerate(self.transitions_buffer):
    #             if not transition.done:
    #                 reward += self.gamma**i * transition.reward
    #             else:
    #                 break
    #         return Transition(
    #             state=self.buffer[0].state,
    #             reward=reward,
    #             next_state=self.buffer[0].next_state,
    #             action=self.buffer[0].action,
    #             done=self.buffer[0].done,
    #         )
    #     else:
    #         return None
    def __init__(self):
        super().__init__()
        
    def write_to_buffer(self, state, reward, next_state, action, done):
        self.data.append(Transition(
            state=state,
            reward=reward,
            next_state=next_state,
            action=action,
            done=done,
        ))        

    def clean(self):
        self.data = []
