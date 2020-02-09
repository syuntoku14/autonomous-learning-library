import torch
from all.nn import weighted_mse_loss
from ._agent import Agent
from tqdm import tqdm
import numpy as np

class DDQfD(Agent):
    '''
    Double Deep Q-Network from demonstration (DDQNfD).
    https://arxiv.org/abs/1704.03732

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        loss (function): The weighted loss function to use.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    '''
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 margin=0.8,
                 expert_epsilon=0.2,
                 initial_training_iters=750000,
                 discount_factor=0.99,
                 loss=weighted_mse_loss,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss = staticmethod(loss)
        # hyperparameters
        self.margin = margin
        self.expert_epsilon = expert_epsilon
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.initial_training_iters = initial_training_iters
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0
        self._should_init_train = True
        
    def act(self, state, reward):
        if self._should_init_train:
            print("initial training starts!")
            for _ in tqdm(range(self.initial_training_iters)):
                self._train()
                self._should_init_train = False
            print("initial training finished!")
        
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self.policy(state)
        return self._action

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            next_actions = torch.argmax(self.q.eval(next_states), dim=1)
            targets = rewards + self.discount_factor * self.q.target(next_states, next_actions)
            # compute Q loss
            q_loss = self.loss(values, targets, weights)

            # margin classification loss
            mc_loss = 0.0
            expert_mask = torch.tensor(np.where(np.array(
                self.replay_buffer._cache) < self.replay_buffer.expert_size, 1, 0)).to(self.q.device)
            num_experts = torch.sum(expert_mask)
            if num_experts > 0:
                values_arr = self.q(states)
                margin_arr = torch.ones(values_arr.shape).to(self.q.device) * self.margin
                for i, margin in enumerate(margin_arr):
                    margin_arr[i][actions[i].item()] = 0
                maxQ_margin, _ = torch.max(values_arr + margin_arr, dim=1)
                mc_loss += torch.sum(expert_mask * (maxQ_margin - values)) / num_experts

            # backward pass
            loss = q_loss + mc_loss
            self.q.reinforce(loss)
            # update replay buffer priorities
            td_errors = targets - values
            priorities = td_errors.abs()
            priorities += expert_mask * self.expert_epsilon
            self.replay_buffer.update_priorities(td_errors.abs())

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0