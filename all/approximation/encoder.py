import torch
from all.nn import RLNetwork
from .approximation import Approximation

class TPIM_Encoder(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='TPIM_Encoder',
            **kwargs
    ):
        model = EncoderModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class EncoderModule(RLNetwork):
    def forward(self, states, actions):
        x = torch.cat((states.features.float(), actions.reshape(-1, 2).float()), dim=1)
        return self.model(x).squeeze(-1)