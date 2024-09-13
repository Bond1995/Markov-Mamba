# Copyright (c) 2024, Albert Gu and Tri Dao.

from torch import nn
from torch.nn import functional as F
import wandb


class GatedMLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = F.relu if config.activation=="relu" else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x, save_weights=False):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)

        if save_weights and self.config.wandb:
            print("fc1-final:")
            print(self.fc1.weight)
            wandb.log({"fc1-final": wandb.Image(self.fc1.weight.numpy(force=True))})

            print("fc2-final:")
            print(self.fc2.weight)
            wandb.log({"fc2-final": wandb.Image(self.fc2.weight.numpy(force=True))})
        
        return y
    
class MLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.activation = F.relu if config.activation=="relu" else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x, save_weights=False):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)

        if save_weights and self.config.wandb:
            print("fc1-final:")
            print(self.fc1.weight)
            wandb.log({"fc1-final": wandb.Image(self.fc1.weight.numpy(force=True))})

            print("fc2-final:")
            print(self.fc2.weight)
            wandb.log({"fc2-final": wandb.Image(self.fc2.weight.numpy(force=True))})
        
        return y
