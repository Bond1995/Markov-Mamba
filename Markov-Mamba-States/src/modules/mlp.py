# Copyright (c) 2024, Albert Gu and Tri Dao.

from torch import nn
from torch.nn import functional as F
import wandb


class GatedMLP(nn.Module):
    def __init__(
        self,
        config,
        id,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=False,
        factor=4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        self.id = id
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features if out_features is not None else in_features
        #hidden_features = (
        #    hidden_features if hidden_features is not None else int(8 * in_features / 3)
        #)
        #hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        hidden_features = hidden_features if hidden_features is not None else factor * in_features
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = F.relu if config.activation=="relu" else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x, save_weights=False):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)

        if save_weights and self.config.wandb:
            print("fc1-l"+str(self.id))
            print(self.fc1.weight)
            wandb.log({"fc1-l"+str(self.id): wandb.Image(self.fc1.weight.numpy(force=True))})

            print("fc2-l"+str(self.id))
            print(self.fc2.weight)
            wandb.log({"fc2-l"+str(self.id): wandb.Image(self.fc2.weight.numpy(force=True))})
        
        return y
    
class MLP(nn.Module):
    def __init__(
        self,
        config,
        id,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=False,
        factor=4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        self.id = id
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features if out_features is not None else in_features
        #hidden_features = (
        #    hidden_features if hidden_features is not None else int(8 * in_features / 3)
        #)
        #hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        hidden_features = hidden_features if hidden_features is not None else factor * in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.activation = F.relu if config.activation=="relu" else F.silu
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x, save_weights=False):
        y = self.fc1(x)

        if save_weights:
            print("After W1:")
            print(y[0,:30])
        y = self.activation(y)
        if save_weights:
            print("After ReLU:")
            print(y[0,:30])
        y = self.fc2(y)
        if save_weights:
            print("After W2:")
            print(y[0,:30])

        if not self.training and self.config.wandb and save_weights:
            print("fc1-l"+str(self.id))
            print(self.fc1.weight)
            wandb.log({"fc1-l"+str(self.id): wandb.Image(self.fc1.weight.numpy(force=True))})

            print("fc2-l"+str(self.id))
            print(self.fc2.weight)
            wandb.log({"fc2-l"+str(self.id): wandb.Image(self.fc2.weight.numpy(force=True))})
        
        return y
