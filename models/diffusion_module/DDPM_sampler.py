from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoiseDiffusion(nn.Module):
    def __init__(self, eps_model, n_steps, device):
        super().__init__()
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
        self.eps_model = eps_model
    

    def q_xt_x0(self, x0, t):
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var
    
    def q_sample(self, x0, t, eps):
        if eps is None:
            eps = torch.rand_like(x0).to(x0.device)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, xt, t, cond):
        eps_theta = self.eps_model(xt, t, cond)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps
    
    def loss(self, x0, noise, cond):
        _ = self.beta + torch.ones_like(self.beta, device=x0.device)
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.rand_like(x0).to(x0.device)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t, cond)
        return F.mse_loss(noise, eps_theta)





