from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import SAC, GAIL_SAC_TPIM
from all.approximation import QContinuous, PolyakTarget, VNetwork, Discriminator, TPIM_Encoder
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies.soft_deterministic import SoftDeterministicPolicy
from all.memory import ExperienceReplayBuffer, HERBuffer
from .models import fc_q, fc_v, fc_soft_policy, fc_discriminator, fc_encoder


def gail_sac_tpim(
        # Common settings
        expert_replay_buffer,
        device="cuda",
        pretrained_model=None,
        discount_factor=0.98,
        last_frame=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_v=1e-3,
        lr_pi=1e-4,
        lr_e=1e-3,
        lr_d=1e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=2,
        polyak_rate=0.005,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6,
        # Exploration settings
        temperature_initial=0.1,
        lr_temperature=1e-5,
        entropy_target_scaling=1.,
        dim_s=32
):
    """
    SAC continuous control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent..
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_q (float): Learning rate for the Q networks.
        lr_v (float): Learning rate for the state-value networks.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        temperature_initial (float): Initial value of the temperature parameter.
        lr_temperature (float): Learning rate for the temperature. Should be low compared to other learning rates.
        entropy_target_scaling (float): The target entropy will be -(entropy_target_scaling * env.action_space.shape[0])
    """
    def _gail_sac_tpim(env, writer=DummyWriter()):
        final_anneal_step = (
            last_frame - replay_start_size) // update_frequency

        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            scheduler=CosineAnnealingLR(
                q_1_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            scheduler=CosineAnnealingLR(
                q_2_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_2'
        )

        v_model = fc_v(env).to(device)
        v_optimizer = Adam(v_model.parameters(), lr=lr_v)
        v = VNetwork(
            v_model,
            v_optimizer,
            scheduler=CosineAnnealingLR(
                v_optimizer,
                final_anneal_step
            ),
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='v',
        )

        # compress the observation
        encoder_model = fc_encoder(env, dim_s).to(device)
        encoder_optimizer = Adam(
            encoder_model.parameters(), lr=lr_e)
        encoder = TPIM_Encoder(encoder_model, encoder_optimizer)

        # policy discriminator takes the compressed vector
        policy_discriminator_model = fc_discriminator(dim_s).to(device)
        policy_d_optimizer = Adam(
            policy_discriminator_model.parameters(), lr=lr_d)
        policy_discriminator = Discriminator(
            policy_discriminator_model, policy_d_optimizer)

        # domain discriminator takes the compressed vector
        domain_discriminator_model = fc_discriminator(dim_s).to(device)
        domain_d_optimizer = Adam(
            domain_discriminator_model.parameters(), lr=lr_d)
        domain_discriminator = Discriminator(
            domain_discriminator_model, domain_d_optimizer)

        # policy takes the compressed vector
        policy_model = fc_soft_policy(env).to(
            device) if pretrained_model is None else pretrained_model
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )
        replay_buffer = HERBuffer(replay_buffer)

        return GAIL_SAC(
            policy,
            q_1,
            q_2,
            v,
            domain_discriminator,
            policy_discriminator,
            encoder,
            replay_buffer,
            expert_replay_buffer,
            temperature_initial=temperature_initial,
            entropy_target=(-env.action_space.shape[0]
                            * entropy_target_scaling),
            lr_temperature=lr_temperature,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
            writer=writer
        )
    return _gail_sac_tpim
