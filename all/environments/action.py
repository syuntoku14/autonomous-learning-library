import numpy as np
import torch
import warnings

class Action:
    def __init__(self, raw):
        """
        Members of Action class:
        1. raw (torch.Tensor): batch_size x shape
        """
        assert isinstance(raw, torch.Tensor), "Input invalid raw type {}. raw must be torch.Tensor".format(type(raw))
        if raw.shape == 1:
            warnings.warn("raw.shape {} is confusing. Should be batched.".format(raw.shape))
            raw = raw.unsqueeze(0)
        self._raw = raw

    @classmethod
    def from_numpy(cls, actions):
        raw = torch.from_numpy(actions)
        return cls(raw)

    @classmethod
    def from_list(cls, actions):
        raw = torch.cat([action.raw for action in actions])
        return cls(raw)

    @property
    def features(self):
        '''
        Default features are the raw state.
        Override this method for other types of features.
        '''
        return self._raw

    @property
    def raw(self):
        return self._raw

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Action(
                self._raw[idx],
            )
        if isinstance(idx, torch.Tensor):
            return Action(
                self._raw[idx],
            )
        return Action(
            self._raw[idx].unsqueeze(0),
        )

    def __len__(self):
        return len(self._raw)