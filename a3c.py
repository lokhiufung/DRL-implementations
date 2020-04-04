####################
# reference: 
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
####################
import argparse
import time
import os

import torch
from torch import nn
# from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim 
import gym
from utils import load_json, get_logger
from tensorboard_logger import TensorboardLogger


logger = get_logger('ac3', fh_lv='debug', ch_lv='debug')


parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str, help='name of experiment')
args = parser.parse_args()

HYPARAMS = load_json('./hyparams/a3c_hyparams.json')[args.name]['hyparams']
logger.debug('hyparams: {}'.format(HYPARAMS))


def ensure_shared_grads(model, shared_model):
    # directly copyied from: https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        # if shared_param.grad is not None:
        #     return
        shared_param._grad = param.grad  # grad is readonly, whereas _grad is writable


def train(shared_model, shared_counter, lock, rank, optimizer=None):
    env = gym.make('CartPole-v0')
    env.seed(HYPARAMS['seed'] + rank)
    gamma = 0.999
    model = ActorCriticNetwork(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n
    )
    counter = 1
    optimizer = optim.RMSprop(shared_model.parameters())
    state = env.reset()  # init the first state

    while True:
        # 0. synchronize model parameters and shared_model parameters
        model.load_state_dict(shared_model.state_dict())
        rewards = []
        value_tensors = []
        prob_tensors = []
        action_tensors = []
        # 1. k steps for accummulating rewards and states
        for step in range(HYPARAMS['max_steps']):
            # forward to get an action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  
            prob_tensor, value_tensor = model(state_tensor)
            prob_tensors.append(prob_tensor)
            value_tensors.append(value_tensor)
            action_tensor = prob_tensor.multinomial(num_samples=1).detach()
            action_tensors.append(action_tensor)
            logger.debug('rank: {} prob: {} value: {} action: {}'.format(rank, prob_tensor, value_tensor, action_tensor))
            # take a step
            next_state, reward, done, info = env.step(action_tensor.item())
            rewards.append(reward)
            counter += 1
            with lock:
                shared_counter.value += 1
            if done:
                state = env.reset()  # reset env from done state
                logger.debug('rank: {} step: {} max_steps: {}'.format(rank, step, HYPARAMS['max_steps']))
                break
            state = next_state  # update the state variable
        # 2. calcute the discounted rewards and loss
        if done:
            R = torch.zeros((1, 1), dtype=torch.float32)
        else:
            _, value_tensor = model(torch.from_numpy(next_state).float().unsqueeze(0))
            R = value_tensor.detach()
        logger.debug('rank: {} R: {} done: {}'.format(rank, R.item(), done))
        value_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0
        for i in reversed(range(len(rewards))):  # rewards, value_tensors, prob_tensors should share the same length
            R = rewards[i] + gamma * R
            advantage = R - value_tensors[i]
            logprob_tensor = torch.log(prob_tensor)
            entropy_loss -= -(logprob_tensor * prob_tensor).sum(1)  # maximize entropy for regularization
            policy_loss -= logprob_tensor.gather(1, action_tensors[i]) * advantage  # gradient ascend
            value_loss += advantage.pow(2)
            logger.debug('rank: {} prob: {} logprob: {} advantage: {}'.format(rank, prob_tensor, logprob_tensor, advantage))
        logger.debug('rank: {} policy_loss: {} entropy_loss: {} value_loss: {}'.format(rank, policy_loss, entropy_loss, value_loss))
        loss = policy_loss + value_loss + HYPARAMS['beta'] * entropy_loss
        with lock:
            # 3. backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # 4. synchronize gradients with the shared_model
            ensure_shared_grads(model, shared_model)

            # 5. asynchronous update 
            optimizer.step()  # This will update the shared parameters


def test(shared_model, shared_counter):
    tensorboard_logdir = './experiment'
    if not os.path.exists(tensorboard_logdir):
        os.mkdir(tensorboard_logdir)

    writer = TensorboardLogger(tensorboard_logdir)
    env = gym.make('CartPole-v0')
    model = ActorCriticNetwork(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n
    )
    model.load_state_dict(shared_model.state_dict())
    model.eval()
    episode = 1
    optimizer = optim.RMSprop(shared_model.parameters())
    state = env.reset()
    state_tensor = tensor.from_numpy(state).unsqueeze(0)  # init the first state
    done = False
    score = 0  # score per episode
    # start = time.perf_counter()
    while True:
        # forward to get an action
        with torch.zero_grad():
            prob_tensor, value_tensor = model(state)
            action_tensor = prob_tensor.multinomial(num_samples=1).detach()
        # take a step
        next_state, reward, done, info = env.step(action_tensor.item())
        rewards.append(reward)
        if done:
            writer.log_episode(episode, score)
            episode += 1
            score = 0
            state = env.reset()  # reset env from done state
            
            time.sleep(30)  # sleep for 30s  
            # update the parameters after every test episode
            model.load_state_dict(shared_model.state_dict())
            break
        score += 1 
        state = next_state  # update the state variable
    writer.close()
        


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.action_output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.value_output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        action = F.softmax(self.action_output_layer(x), dim=-1)
        value = self.value_output_layer(x)
        return action, value

if __name__ == '__main__':
    # temp: experiment name
    hyparams = load_json('./hyparams/a3c_hyparams.json')['a3c-01']['hyparams']
    num_actors = hyparams['num_actors']

    shared_model = ActorCriticNetwork(input_dim=4, output_dim=2, hidden_dim=128)
    # NOTE: this is required for the ``fork`` method to work
    shared_model.share_memory()

    processes = []

    shared_counter = mp.Value('i', 0)
    lock = mp.Lock()

    # train(shared_model, shared_counter, lock, rank)
    p = mp.Process(target=test, args=(shared_model, shared_counter))
    p.start()
    processes.append(p)
    for rank in range(HYPARAMS['num_actors']):
        p = mp.Process(target=train, args=(shared_model, shared_counter, lock, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()