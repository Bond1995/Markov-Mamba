# Copyright (c) 2024, Albert Gu and Tri Dao.

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.mamba2 import Mamba2
from modules.mlp import GatedMLP, MLP


class Block(nn.Module):
    def __init__(
        self, config
    ):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.device = config.device
        self.dtype = config.dtype
        activation = F.relu if config.activation == "relu" else F.silu
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.mixer = Mamba2(config, **factory_kwargs)
        if self.config.layernorm:
            self.norm = nn.LayerNorm(self.d_model, bias=False, **factory_kwargs)
            self.norm2 = nn.LayerNorm(self.d_model, bias=False, **factory_kwargs)
        if self.config.gate_act:
            self.mlp = GatedMLP(self.d_model, activation=activation, **factory_kwargs)
        else:
            self.mlp = MLP(self.d_model, activation=activation, **factory_kwargs)

    def forward(self, hidden_states):
        residual = hidden_states
        if self.config.layernorm:
            hidden_states = self.norm(hidden_states).to(self.dtype)
        hidden_states = self.mixer(hidden_states)

        residual = hidden_states + residual
        if self.config.layernorm:
            hidden_states = self.norm2(residual).to(self.dtype)
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(residual)

        return hidden_states + residual


class MambaLMHeadModel(nn.Module):

    def __init__(
        self,
        config
    ) -> None:
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList([Block(config) for i in range(n_layer)])
        if self.config.layernorm:
            self.norm_f = nn.LayerNorm(d_model, bias=False, **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)
        self.tie_weights()
    
    def _init_weights(
        self,
        module,
        initializer_range=0.02,
        n_residuals_per_layer=2,
    ):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * self.config.n_layer)
    
    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def get_parameter_group_specs(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.Conv1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if hasattr(p, '_no_weight_decay'):
                    if p._no_weight_decay == True:
                        no_decay.add(fpn)
                elif pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.remove('lm_head.weight')
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
    
    def forward(self, idx, targets):
        hidden_states = self.embedding(idx)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        if self.config.layernorm:
            hidden_states = self.norm_f(hidden_states).to(self.dtype)
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return {'logits': logits, 'loss': loss}