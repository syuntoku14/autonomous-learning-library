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

    def expert_reward(self, features, next_features):
        return -torch.log(self.model(features, next_features)).detach()

class DiscriminatorModule(RLNetwork):
    def forward(self, features, next_features):
        return self.model(features, next_features)