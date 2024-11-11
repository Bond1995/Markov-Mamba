# Copyright (c) 2024, Albert Gu and Tri Dao.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from einops import rearrange

from modules.ssd_minimal import ssd_minimal_discrete
from models.mamba_llm import compute_energies

class Mamba2(nn.Module):
    def __init__(
        self,
        config,
        id,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.id = id
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = self.expand * self.d_model
        self.nheads = config.nheads
        self.ngroups = config.ngroups
        assert self.d_inner % self.nheads == 0
        self.headdim = self.d_inner // self.nheads
        self.dt_limit = dt_limit
        self.activation = config.activation
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        if self.config.conv:
            if self.config.conv_type == "onlyx":
                conv_dim = self.d_inner
                self.conv1d = nn.Conv1d(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    bias=conv_bias,
                    kernel_size=self.d_conv,
                    groups=conv_dim,
                    padding=self.d_conv - 1,
                    **factory_kwargs,
                )
            elif self.config.conv_type == "onlyb" or self.config.conv_type == "onlyc":
                conv_dim = self.ngroups * self.d_state
                self.conv1d = nn.Conv1d(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    bias=conv_bias,
                    kernel_size=self.d_conv,
                    groups=conv_dim,
                    padding=self.d_conv - 1,
                    **factory_kwargs,
                )
            else:
                conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
                self.conv1d = nn.Conv1d(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    bias=conv_bias,
                    kernel_size=self.d_conv,
                    groups=conv_dim,
                    padding=self.d_conv - 1,
                    **factory_kwargs,
                )

        if self.config.conv_act:
            if self.activation == "relu":
                self.act = nn.ReLU()
            else:
                self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter (removed since it seems not to be there in the paper)
        self.D = None
        # self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        # self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        if self.config.layernorm:
            self.norm = nn.LayerNorm(self.d_inner, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, save_weights=False):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        if save_weights:
            print("Input to Mamba:")
            print(u[0,:30])
            if self.config.wandb:
                wandb.log({"u-l"+str(self.id): wandb.Image(u[0,:30].numpy(force=True).squeeze())})
        
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log).to(self.dtype)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias).to(self.dtype)  # (B, L, nheads)
        assert self.activation in ["silu", "relu"]

        # 1D Convolution
        if self.config.conv:
            if self.config.conv_type == "onlyx":
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    x = self.act(x)
                x = x[:, :seqlen, :].to(self.dtype)
                B = B[:, :seqlen, :].to(self.dtype)
                C = C[:, :seqlen, :].to(self.dtype)
            elif self.config.conv_type == "onlyb":
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                B = self.conv1d(B.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    B = self.act(B)
                x = x[:, :seqlen, :].to(self.dtype)
                B = B[:, :seqlen, :].to(self.dtype)
                C = C[:, :seqlen, :].to(self.dtype)
            elif self.config.conv_type == "onlyc":
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                C = self.conv1d(C.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    C = self.act(C)
                x = x[:, :seqlen, :].to(self.dtype)
                B = B[:, :seqlen, :].to(self.dtype)
                C = C[:, :seqlen, :].to(self.dtype)
            else:
                xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    xBC = self.act(xBC)
                xBC = xBC[:, :seqlen, :].to(self.dtype) # (B, L, self.d_inner + 2 * ngroups * d_state)
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        if save_weights:
            print("X, B, C after convolution:")
            print(x[0,:30])
            print(B[0,:30])
            print(C[0,:30])
            if self.config.wandb:
                wandb.log({"cx-l"+str(self.id): wandb.Image(x[0,:30].numpy(force=True).squeeze())})
                wandb.log({"cB-l"+str(self.id): wandb.Image(B[0,:30].numpy(force=True).squeeze())})
                wandb.log({"cC-l"+str(self.id): wandb.Image(C[0,:30].numpy(force=True).squeeze())})

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        y, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, self.chunk_size)
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        if self.config.layernorm:
            y = self.norm(y)
        if self.config.gate:
            if self.activation == "relu":
                z = F.relu(z)
            else:
                z = F.silu(z)
            y = y * z
        
        out = self.out_proj(y)

        if self.config.wandb:
            if save_weights:
                print("in_proj-l"+str(self.id))
                print(self.in_proj.weight)
                print(compute_energies(self.in_proj.weight.numpy(force=True)))
                wandb.log({"in_proj-l"+str(self.id): wandb.Image(self.in_proj.weight.numpy(force=True))})

                print("out_proj-l"+str(self.id))
                print(self.out_proj.weight)
                print(compute_energies(self.out_proj.weight.numpy(force=True)))
                wandb.log({"out_proj-l"+str(self.id): wandb.Image(self.out_proj.weight.numpy(force=True))})

                if self.config.conv:
                    print("conv-l"+str(self.id))
                    print(self.conv1d.weight)
                    wandb.log({"conv-l"+str(self.id): wandb.Image(self.conv1d.weight.numpy(force=True).squeeze())})
                    print("conv-bias-l"+str(self.id))
                    print(self.conv1d.bias)
            if self.training and self.nheads == 1:
                wandb.log({
                    "params/A-l"+str(self.id): torch.exp(self.A_log).item(),
                    "params/dt_bias-l"+str(self.id): self.dt_bias.item(),
                })

        return out
