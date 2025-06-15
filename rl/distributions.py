import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.kl import kl_divergence


class MultiCategorical(Distribution):

    def __init__(self, multi_logits, nvec, validate_args=False):
        self.cats = [
            Categorical(logits=logits)
            for logits in torch.split(multi_logits, nvec, dim=-1)
        ]
        batch_shape = multi_logits.size()[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self) -> Tensor:
        return torch.stack([cat.sample() for cat in self.cats], dim=-1)

    def mode(self) -> Tensor:
        return torch.stack([cat.mode for cat in self.cats], dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        value = torch.unbind(value, dim=-1)
        logps = torch.stack([cat.log_prob(act) for cat, act in zip(self.cats, value)])
        return torch.sum(logps, dim=0)

    def entropy(self) -> Tensor:
        return torch.stack([cat.entropy() for cat in self.cats], dim=-1).sum(dim=-1)

    def kl(self, other):
        kls = torch.stack(
            [
                kl_divergence(cat, oth_cat)
                for cat, oth_cat in zip(self.cats, other.cats)
            ],
            dim=-1,
        )
        return torch.sum(kls, dim=-1)
