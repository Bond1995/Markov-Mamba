# Copyright (c) 2024, Albert Gu and Tri Dao.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

from einops import rearrange

from modules.ssd_minimal import ssd_minimal_discrete


def compute_energies(W):
    sv = torch.linalg.svdvals(W)
    energies = torch.cumsum(sv, dim=0)
    energies = energies / energies[-1]

    return energies

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
            if self.config.conv_type == "fixed":
                std = math.sqrt(1/self.d_conv)
                ker = torch.rand(1, self.d_conv) * 2 * std - std
                self.ker = nn.Parameter(ker)
            elif self.config.conv_type == "onlyx":
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
            elif self.config.conv_type == "onlyxb":
                conv_dim = self.d_inner + self.ngroups * self.d_state
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

    def forward(self, u, save_weights=False, check_conditions=False):
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

        if check_conditions:
            if self.config.conv_type == "onlyxb":
                xB, _ = torch.split(xBC, [self.d_inner + self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                xB0 = xB[0, 0].unsqueeze(0)
                xB0 = self.conv1d(xB0.transpose(0, 1)).transpose(0, 1)
                xB0 = xB0[0].to(self.dtype)
                x0, b0 = torch.split(xB0, [self.d_inner, self.ngroups * self.d_state], dim=-1)
                x0 = x0 * dt[0, 0, 0]
                xB1 = xB[0, 2].unsqueeze(0)
                xB1 = self.conv1d(xB1.transpose(0, 1)).transpose(0, 1)
                xB1 = xB1[0].to(self.dtype)
                x1, b1 = torch.split(xB1, [self.d_inner, self.ngroups * self.d_state], dim=-1)
                x1 = x1 * dt[0, 2, 0]
            else:
                xBC0 = xBC[0, 0].unsqueeze(0)
                xBC0 = self.conv1d(xBC0.transpose(0, 1)).transpose(0, 1)
                xBC0 = xBC0[0].to(self.dtype)
                x0, b0, c0 = torch.split(xBC0, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                x0 = x0 * dt[0, 0, 0]
                xBC1 = xBC[0, 2].unsqueeze(0)
                xBC1 = self.conv1d(xBC1.transpose(0, 1)).transpose(0, 1)
                xBC1 = xBC1[0].to(self.dtype)
                x1, b1, c1 = torch.split(xBC1, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                x1 = x1 * dt[0, 2, 0]

        # 1D Convolution
        if self.config.conv:
            if self.config.conv_type == "fixed":
                xBC = F.conv1d(
                    xBC.transpose(1, 2), 
                    self.ker.repeat(self.d_inner + 2 * self.ngroups * self.d_state, 1, 1),
                    groups=self.d_inner + 2 * self.ngroups * self.d_state,
                    padding=self.d_conv - 1).transpose(1, 2)
                if self.config.conv_act:
                    xBC = self.act(xBC)
                xBC = xBC[:, :seqlen, :].to(self.dtype) # (B, L, self.d_inner + 2 * ngroups * d_state)
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            elif self.config.conv_type == "onlyx":
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    x = self.act(x)
                x = x[:, :seqlen, :].to(self.dtype)
                B = B[:, :seqlen, :].to(self.dtype)
                C = C[:, :seqlen, :].to(self.dtype)
            elif self.config.conv_type == "onlyxb":
                xB, C = torch.split(xBC, [self.d_inner + self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
                xB = self.conv1d(xB.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    xB = self.act(xB)
                x, B = torch.split(xB, [self.d_inner, self.ngroups * self.d_state], dim=-1)
                x = x[:, :seqlen, :].to(self.dtype)
                B = B[:, :seqlen, :].to(self.dtype)
                C = C[:, :seqlen, :].to(self.dtype)
            else:
                xBC = self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                if self.config.conv_act:
                    xBC = self.act(xBC)
                xBC = xBC[:, :seqlen, :].to(self.dtype) # (B, L, self.d_inner + 2 * ngroups * d_state)
                x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        else:
            if self.config.conv_act:
                xBC = self.act(xBC)
            x, B, C = torch.split(xBC.to(self.dtype), [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        if save_weights:
            print("X, B, C, dt after convolution:")
            print(x[0,:30])
            print(B[0,:30])
            print(C[0,:30])
            print(dt[0,:30])
            if self.config.wandb:
                wandb.log({"cx-l"+str(self.id): wandb.Image(x[0,:30].numpy(force=True).squeeze())})
                wandb.log({"cB-l"+str(self.id): wandb.Image(B[0,:30].numpy(force=True).squeeze())})
                wandb.log({"cC-l"+str(self.id): wandb.Image(C[0,:30].numpy(force=True).squeeze())})

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        if save_weights:
            At = A*dt
            print("A_t:")
            print(At[0, :30, 0])
            np.save('At', At.numpy(force=True))
            if self.config.wandb:
                wandb.save('At.npy')
        
        if check_conditions:
            if self.config.conv_type == "onlyxb":
                # Compute vectors
                xdt = x*dt.unsqueeze(-1)
                x00 = xdt[0,1,0]
                x01 = xdt[0,2,0]
                x10 = xdt[0,4,0]
                x11 = xdt[0,3,0]
                b00 = B[0,1,0]
                b01 = B[0,2,0]
                b10 = B[0,4,0]
                b11 = B[0,3,0]
                c0 = C[0,1,0]
                c1 = C[0,2,0]
                # Print vectors
                print("Starting vectors:")
                print(x0,x1,b0,b1, sep='\n')
                print("X vectors:")
                print(x00,x01,x10,x11, sep='\n')
                print("B vectors:")
                print(b00,b01,b10,b11, sep='\n')
                print("C vectors:")
                print(c0,c1, sep='\n')
                # Check inner products
                print("Inner products:")
                print(c0 @ b00)
                print(c0 @ b01)
                print(c0 @ b10)
                print(c0 @ b11)
                print(c1 @ b00)
                print(c1 @ b01)
                print(c1 @ b10)
                print(c1 @ b11)
            else:
                # Compute vectors
                xdt = x*dt.unsqueeze(-1)
                x00 = xdt[0,1,0]
                x01 = xdt[0,2,0]
                x10 = xdt[0,4,0]
                x11 = xdt[0,3,0]
                b00 = B[0,1,0]
                b01 = B[0,2,0]
                b10 = B[0,4,0]
                b11 = B[0,3,0]
                c00 = C[0,1,0]
                c01 = C[0,2,0]
                c10 = C[0,4,0]
                c11 = C[0,3,0]
                # Print vectors
                print("Starting vectors:")
                print(x0,x1,b0,b1,c0,c1, sep='\n')
                print("X vectors:")
                print(x00,x01,x10,x11, sep='\n')
                print("B vectors:")
                print(b00,b01,b10,b11, sep='\n')
                print("C vectors:")
                print(c00,c01,c10,c11, sep='\n')
                # Check inner products
                print("Inner products:")
                print(c00 @ b11)
                print(c10 @ b11)
                print(c01 @ b00)
                print(c11 @ b00)
                print("------")
                print(c00 @ b00)
                print(c11 @ b11)

        if self.config.fix_A:
            y, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), 0.0*dt, B, C, self.chunk_size)
        else:
            y, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, self.chunk_size)
        y = rearrange(y, "b l h p -> b l (h p)")

        if save_weights:
            print("y:")
            print(y[0,:30])
        
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
        out = y

        if save_weights:
            print("out:")
            print(out[0,:30])
            
            if self.config.wandb:
                dz = self.d_inner
                dx = 2 * self.d_inner
                db = 2 * self.d_inner + self.ngroups * self.d_state
                dc = 2 * self.d_inner + 2 * self.ngroups * self.d_state
                dd = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

                print("Wz-l"+str(self.id))
                print(self.in_proj.weight[:dz,:])
                print(compute_energies(self.in_proj.weight[:dz,:].numpy(force=True)))
                wandb.log({"Wz-l"+str(self.id): wandb.Image(self.in_proj.weight[:dz,:].numpy(force=True))})

                print("Wx-l"+str(self.id))
                print(self.in_proj.weight[dz:dx,:])
                print(compute_energies(self.in_proj.weight[dz:dx,:].numpy(force=True)))
                wandb.log({"Wx-l"+str(self.id): wandb.Image(self.in_proj.weight[dz:dx,:].numpy(force=True))})

                print("Wb-l"+str(self.id))
                print(self.in_proj.weight[dx:db,:])
                print(compute_energies(self.in_proj.weight[dx:db,:].numpy(force=True)))
                wandb.log({"Wb-l"+str(self.id): wandb.Image(self.in_proj.weight[dx:db,:].numpy(force=True))})

                print("Wc-l"+str(self.id))
                print(self.in_proj.weight[db:dc,:])
                print(compute_energies(self.in_proj.weight[db:dc,:].numpy(force=True)))
                wandb.log({"Wc-l"+str(self.id): wandb.Image(self.in_proj.weight[db:dc,:].numpy(force=True))})

                print("Wdelta-l"+str(self.id))
                print(self.in_proj.weight[dc:dd,:])
                print(compute_energies(self.in_proj.weight[dc:dd,:].numpy(force=True)))
                wandb.log({"Wdelta-l"+str(self.id): wandb.Image(self.in_proj.weight[dc:dd,:].numpy(force=True))})

                print("out_proj-l"+str(self.id))
                print(self.out_proj.weight)
                print(compute_energies(self.out_proj.weight.numpy(force=True)))
                wandb.log({"out_proj-l"+str(self.id): wandb.Image(self.out_proj.weight.numpy(force=True))})

                if self.config.conv:
                    if self.config.conv_type == "fixed":
                        print("conv-l"+str(self.id))
                        print(self.ker)
                    else:
                        print("conv-l"+str(self.id))
                        print(self.conv1d.weight)
                        wandb.log({"conv-l"+str(self.id): wandb.Image(self.conv1d.weight.numpy(force=True).squeeze())})
                        print("conv-bias-l"+str(self.id))
                        print(self.conv1d.bias)
            if not self.training and not save_weights and not check_conditions and self.nheads == 1:
                wandb.log({
                    "params/A-l"+str(self.id): torch.exp(self.A_log).item(),
                    "params/dt_bias-l"+str(self.id): self.dt_bias.item(),
                })

        return out
