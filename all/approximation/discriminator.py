import torch
from all.nn import RLNetwork
from .approximation import Approximation

class Discriminator(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='discriminator',
            **kwargs
    ):
        model = DiscriminatorModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def expert_reward(self, state, next_states):
        return -torch.log(self.model(state, next_states)).detach()

class DiscriminatorModule(RLNetwork):
    def forward(self, states, next_states):
        x = torch.cat((states.features.float(), next_states.features.float()), dim=1)
        return self.model(x).squeeze(-1) * states.mask.float()