import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from torch.nn import BCELoss
from ._agent import Agent
from pytorch_revgrad.functional import revgrad


class GAIL_SAC(Agent):
    """
    Soft Actor-Critic (SAC).
    SAC is a proposed improvement to DDPG that replaces the standard
    mean-squared Bellman error (MSBE) objective with a "maximum entropy"
    objective that impoves exploration. It also uses a few other tricks,
    such as the "Clipped Double-Q Learning" trick introduced by TD3.
    This implementation uses automatic temperature adjustment to replace the
    difficult to set temperature parameter with a more easily tuned
    entropy target parameter.
    https://arxiv.org/abs/1801.01290

    Args:
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        q1 (QContinuous): An Approximation of the continuous action Q-function.
        q2 (QContinuous): An Approximation of the continuous action Q-function.
        v (VNetwork): An Approximation of the state-value function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        entropy_target (float): The desired entropy of the policy. Usually -env.action_space.shape[0]
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        temperature_initial (float): The initial temperature used in the maximum entropy objective.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(self,
                 policy,
                 q_1,
                 q_2,
                 v,
                 domain_discriminator,
                 policy_discriminator,
                 encoder,
                 replay_buffer,
                 expert_replay_buffer,
                 expert_rew_ratio=0.3,
                 discount_factor=0.99,
                 entropy_target=-2.,
                 lr_temperature=1e-4,
                 minibatch_size=64,
                 replay_start_size=5000,
                 domain_weight=0.1,
                 temperature_initial=0.1,
                 update_frequency=1,
                 writer=DummyWriter()
                 ):
        # objects
        self.policy = policy
        self.v = v
        self.q_1 = q_1
        self.q_2 = q_2
        self.domain_discriminator = domain_discriminator
        self.policy_discriminator = policy_discriminator
        self.encoder = encoder
        self.discrim_criterion = BCELoss()
        self.replay_buffer = replay_buffer
        self.expert_replay_buffer = expert_replay_buffer
        self.writer = writer
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_target = entropy_target
        self.lr_temperature = lr_temperature
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.temperature = temperature_initial
        self.update_frequency = update_frequency
        self.expert_rew_ratio = expert_rew_ratio
        self.domain_weight = domain_weight
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state, reward):
        self.replay_buffer.store(self._state, self._action, reward, state)
        self._train()
        self._state = state
        self._action = self.policy.eval(state)[0]
        return self._action

    def _train(self):
        if self._should_train():
            # sample from replay buffer
            (states, actions, _rewards, next_states,
             _) = self.replay_buffer.sample(self.minibatch_size)
            (ex_states, ex_actions, ex_rewards, ex_next_states,
             _) = self.expert_replay_buffer.sample(self.minibatch_size)

            feature = self.encoder(states, actions)
            ex_feature = self.encoder(ex_states, ex_actions)

            # gail policy disc update
            policy_fake = self.policy_discriminator(feature)
            policy_real = self.policy_discriminator(ex_feature)
            policy_discrim_loss = self.discrim_criterion(policy_fake, torch.ones_like(policy_fake)) + \
                self.discrim_criterion(
                    policy_real, torch.zeros_like(policy_real))

            print("policy_fake: {}, policy_real: {}", policy_fake.mean(), policy_real.mean())
            # gail domain disc update
            domain_fake = self.domain_discriminator(revgrad(feature))
            domain_real = self.domain_discriminator(revgrad(ex_feature))
            domain_discrim_loss = self.discrim_criterion(domain_fake, torch.ones_like(domain_fake)) + \
                self.discrim_criterion(
                    domain_real, torch.zeros_like(domain_real))
            print("domain_fake: {}, domain_real: {}".format(domain_fake.mean().item(), domain_real.mean().item()))

            discrim_loss = self.domain_weight * domain_discrim_loss + policy_discrim_loss
            self.encoder.reinforce(discrim_loss, retain_graph=True)
            self.policy_discriminator.reinforce(
                discrim_loss, retain_graph=True)
            self.domain_discriminator.reinforce(discrim_loss)

            # gail rewards
            expert_rewards = self.policy_discriminator.expert_reward(feature)
            rewards = (1 - self.expert_rew_ratio) * _rewards + \
                self.expert_rew_ratio * expert_rewards

            # compute targets for Q and V
            _actions, _log_probs = self.policy.eval(states)
            q_targets = rewards + self.discount_factor * \
                self.v.target(next_states)
            v_targets = torch.min(
                self.q_1.target(states, _actions),
                self.q_2.target(states, _actions),
            ) - self.temperature * _log_probs

            # update Q and V-functions
            self.q_1.reinforce(mse_loss(self.q_1(states, actions), q_targets))
            self.q_2.reinforce(mse_loss(self.q_2(states, actions), q_targets))
            self.v.reinforce(mse_loss(self.v(states), v_targets))

            # update policy
            _actions2, _log_probs2 = self.policy(states)
            loss = (-self.q_1(states, _actions2) +
                    self.temperature * _log_probs2).mean()
            self.policy.reinforce(loss)
            self.q_1.zero_grad()

            # adjust temperature
            temperature_grad = (_log_probs + self.entropy_target).mean()
            self.temperature += self.lr_temperature * temperature_grad.detach()

            # additional debugging info
            self.writer.add_loss('policy_discrim_loss', policy_discrim_loss.mean())
            self.writer.add_loss('domain_discrim_loss', domain_discrim_loss.mean())
            self.writer.add_loss('entropy', -_log_probs.mean())
            self.writer.add_loss('v_mean', v_targets.mean())
            self.writer.add_loss('r_mean', rewards.mean())
            self.writer.add_scalar('expert_reward', expert_rewards.mean())
            self.writer.add_loss('temperature_grad', temperature_grad)
            self.writer.add_loss('temperature', self.temperature)

    def _should_train(self):
        self._frames_seen += 1
        return len(self.replay_buffer) > self.replay_start_size and self._frames_seen % self.update_frequency == 0
