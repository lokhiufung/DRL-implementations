import random


def take_agent_step(method):
    def wrapper(ref, *args, **kwargs):
        ref.agent_steps += 1
        return method(ref, *args, **kwargs)
    return wrapper


def epsilon_greedy_play_step(play_step):
    def wrapper(ref, *args, **kwargs):
        if ref._exploration_scheduler is not None and hasattr(ref, 'random_play_step'):
            if random.random() > ref._exploration_scheduler.eps:
                return play_step(ref, *args, **kwargs)
            else:
                return ref.random_play_step()
    return wrapper
