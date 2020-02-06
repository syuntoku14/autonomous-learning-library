from abc import ABC, abstractmethod
import numpy as np
import torch
from all.environments import State
from all.optim import Schedulable
from .segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, states, actions, rewards, next_states):
        '''Store the transition in the buffer'''

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''


# Adapted from:
# https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
class ExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, size, device=torch.device('cpu')):
        self.buffer = []
        self.capacity = int(size)
        self.pos = 0
        self.device = device

    def store(self, states, actions, rewards, next_states):
        if states is None or actions is None:
            return
        if len(states) == 1:
            actions = actions.reshape(-1)
            rewards = torch.FloatTensor([rewards])
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            if state is not None and not state.done:
                self._add((state, action, reward, next_state))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return self._reshape(minibatch, torch.ones(batch_size, device=self.device))

    def update_priorities(self, td_errors):
        pass

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def _reshape(self, minibatch, weights):
        states = State.from_list([sample[0] for sample in minibatch]).to(self.device)
        actions = torch.tensor([sample[1] for sample in minibatch], device=self.device)
        rewards = torch.tensor([sample[2] for sample in minibatch], device=self.device).float()
        next_states = State.from_list([sample[3] for sample in minibatch]).to(self.device)
        return (states, actions, rewards, next_states, weights)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

class PrioritizedReplayBuffer(ExperienceReplayBuffer, Schedulable):
    def __init__(
            self,
            buffer_size,
            alpha=0.6,
            beta=0.4,
            epsilon=1e-5,
            device=torch.device('cpu')
    ):
        super().__init__(buffer_size, device=device)

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._beta = beta
        self._epsilon = epsilon
        self._cache = None

    def store(self, states, actions, rewards, next_states):
        if states is None or actions is None:
            return
        if len(states) == 1:
            actions = actions.reshape(-1)
            rewards = torch.FloatTensor([rewards])
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            if state is None or state.done:
                continue
            idx = self.pos
            super()._add((state, action, reward, next_state))
            self._it_sum[idx] = self._max_priority ** self._alpha
            self._it_min[idx] = self._max_priority ** self._alpha

    def sample(self, batch_size):
        beta = self._beta
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        try:
            samples = [self.buffer[idx] for idx in idxes]
        except IndexError as e:
            print('index out of range: ', idxes)
            raise e
        self._cache = idxes
        return self._reshape(samples, torch.from_numpy(weights).to(self.device))

    def update_priorities(self, priorities):
        idxes = self._cache
        _priorities = priorities.detach().cpu().numpy()
        _priorities = np.maximum(_priorities, self._epsilon)
        assert len(idxes) == len(_priorities)
        for idx, priority in zip(idxes, _priorities):
            assert priority > 0
            assert priority < np.inf
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

class NStepReplayBuffer(ReplayBuffer):
    '''Converts any ReplayBuffer into an NStepReplayBuffer'''
    def __init__(
            self,
            steps,
            discount_factor,
            buffer,
    ):
        assert steps >= 1
        assert discount_factor >= 0
        self.steps = steps
        self.discount_factor = discount_factor
        self.buffer = buffer
        self._statess = None
        self._actionss = None
        self._rewardss = None
        self._rewards = None

    def store(self, states, actions, rewards, next_states):
        if states is None or actions is None:
            return

        if len(states) == 1:
            actions = actions.reshape(-1)
            rewards = torch.FloatTensor([rewards])

        # initialize by number of envs
        if self._statess is None:
            self._statess = [[] for _ in range(len(states))]
            self._actionss = [[] for _ in range(len(states))]
            self._rewardss = [[] for _ in range(len(states))]
            self._rewards = [0. for _ in range(len(states))]

        for env_id, (state, action, reward, next_state) in enumerate(zip(states, actions, rewards, next_states)):
            if state is None or state.done:
                continue
            self._statess[env_id].append(state)
            self._actionss[env_id].append(action)
            self._rewardss[env_id].append(reward)
            self._rewards[env_id] += (self.discount_factor ** (len(self._statess[env_id]) - 1)) * rewards[env_id]

            if len(self._statess[env_id]) == self.steps:
                self._store_next(env_id, next_state)

            if not next_state.mask[0]:
                while self._statess[env_id]:
                    self._store_next(env_id, next_state)
                self._rewards[env_id] = 0.

    def _store_next(self, env_id, next_state):
        self.buffer.store(self._statess[env_id][0], self._actionss[env_id][0],
            self._rewards[env_id], next_state)
        self._rewards[env_id] = self._rewards[env_id] -  self._rewardss[env_id][0]
        self._rewards[env_id] *= self.discount_factor ** -1
        del self._statess[env_id][0]
        del self._actionss[env_id][0]
        del self._rewardss[env_id][0]

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)
