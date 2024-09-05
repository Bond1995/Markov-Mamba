# Copyright (c) 2024, Albert Gu and Tri Dao.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from modules.ssd_minimal import ssd_minimal_discrete


class Mamba2(nn.Module):
    def __init__(
        self,
        config,
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
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = self.expand * self.d_model
        self.headdim = config.headdim
        self.ngroups = config.ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
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

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log).to(self.dtype)  # (nheads) or (d_inner, d_state)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias).to(self.dtype)  # (B, L, nheads)
        assert self.activation in ["silu", "relu"]

        # 1D Convolution
        if self.config.conv:
            if self.config.conv_act:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )
            else:
                xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        else:
            if self.config.conv_act:
                xBC = self.act(xBC)
        xBC = xBC[:, :seqlen, :].to(self.dtype) # (B, L, self.d_inner + 2 * ngroups * d_state)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        y, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, self.chunk_size)
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        if self.config.gate_act:
            if self.activation == "relu":
                z = F.relu(z)
            else:
                z = F.silu(z)
        if self.config.layernorm:
            y = self.norm(y) * z
        else:
            y = y * z
        out = self.out_proj(y)

        return out
