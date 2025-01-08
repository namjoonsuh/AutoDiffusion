import functools
import torch
import torch.nn as nn
import torch.nn.functional as F  # For reglu and geglu definitions if used
import numpy as np
import tqdm.notebook
import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from torch import Tensor

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from scipy import integrate

################################################################################
# Dynamically choose device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. Could be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # Ensure timesteps is on the correct device
    timesteps = timesteps.to(device)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )

def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )

class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting]."""

    class Block(nn.Module):
        """The main building block of `MLP`."""
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ['ReGLU', 'GEGLU']

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP` with ReLU activations."""
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'If d_layers contains more than two elements, then'
                ' all elements except for the first and the last must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

################################################################################

# Updated drift_coeff, diffusion_coeff, etc. so they operate on GPU

def drift_coeff(x, t, beta_1, beta_0):
    # ensure t is on same device as x
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device)
    else:
        t = t.to(x.device)
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift = -0.5 * beta_t * x
    return drift

def diffusion_coeff(t, beta_1, beta_0):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device)
    beta_t = beta_0 + t * (beta_1 - beta_0)
    diffusion = torch.sqrt(beta_t)
    return diffusion

def marginal_prob_mean(x, t, beta_1, beta_0):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device)
    else:
        t = t.to(x.device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff)[:, None] * x
    return mean

def marginal_prob_std(t, beta_1, beta_0):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    std = 1 - torch.exp(2. * log_mean_coeff)
    return torch.sqrt(std)

drift_coeff_fn = functools.partial(drift_coeff, beta_1=20, beta_0=0.1)
diffusion_coeff_fn = functools.partial(diffusion_coeff, beta_1=20, beta_0=0.1)
marginal_prob_mean_fn = functools.partial(marginal_prob_mean, beta_1=20, beta_0=0.1)
marginal_prob_std_fn = functools.partial(marginal_prob_std, beta_1=20, beta_0=0.1)

def min_max_scaling(factor, scale=(0, 1)):
    factor = factor.to(device)
    std = (factor - factor.min()) / (factor.max() - factor.min())
    new_min = torch.tensor(scale[0], device=device)
    new_max = torch.tensor(scale[1], device=device)
    return std * (new_max - new_min) + new_min

def compute_v(ll, alpha, beta):
    v = -torch.ones(ll.shape, device=ll.device)
    v[torch.gt(ll, beta)] = 0.
    v[torch.le(ll, alpha)] = 1.

    mask = torch.eq(v, -1)
    if mask.sum() not in [0, 1]:
        v[mask] = min_max_scaling(ll[mask], scale=(1, 0)).to(v.device)
    else:
        v[mask] = 0.5
    return v

def loss_fn(model, Input_Data, T, eps=1e-5):
    """
    model: the score model (on device)
    Input_Data: [N, input_dim], already on device
    T: number of time steps
    """
    N, input_dim = Input_Data.shape
    loss_values = torch.empty(N, device=Input_Data.device)

    for row in range(N):
        # Random times on GPU
        random_t = torch.rand(T, device=Input_Data.device) * (1. - eps) + eps

        # Compute Perturbed data from SDE
        mean = marginal_prob_mean_fn(Input_Data[row, :], random_t)
        std = marginal_prob_std_fn(random_t)
        z = torch.randn(T, input_dim, device=Input_Data.device)
        perturbed_data = mean + z * std[:, None]

        score = model(perturbed_data, random_t)
        loss_row = torch.mean(torch.sum((score * std[:, None] + z) ** 2, dim=1))
        loss_values[row] = loss_row

    return loss_values

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, rtdl_params, dim_t=128):
        super().__init__()
        self.dim_t = dim_t

        # Adjust RTDL MLP parameters to produce an output of dimension d_in
        rtdl_params['d_in'] = dim_t  # input dimension for MLP
        rtdl_params['d_out'] = d_in  # output dimension from MLP

        self.mlp = MLP.make_baseline(**rtdl_params)
        
        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, timesteps):
        """
        x: [batch_size, d_in]
        timesteps: [batch_size] (or [T]) - time steps
        """
        # embed time
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        # project input + add time embedding
        x = self.proj(x) + emb
        return self.mlp(x)

def train_diffusion(latent_features, T, eps, sigma, lr,
                    num_batches_per_epoch, maximum_learning_rate,
                    weight_decay, n_epochs, batch_size):
    """
    :param latent_features: data to train on. Make sure it's a tensor.
    :param T: number of time steps in loss_fn
    :param eps: small epsilon for random_t
    :param sigma: unused in this snippet, but presumably for noise scaling
    ...
    """
    # 1) Move the data to the chosen device
    latent_features = latent_features.to(device)

    # 2) Create the model and place it on the GPU
    rtdl_params = {
        'd_in': latent_features.shape[1],  # Overwritten in MLPDiffusion init
        'd_layers': [256, 256],
        'dropout': 0.0,
        'd_out': latent_features.shape[1]  # Overwritten in MLPDiffusion init
    }
        
    ScoreNet = MLPDiffusion(latent_features.shape[1], rtdl_params)
    ScoreNet_Parallel = torch.nn.DataParallel(ScoreNet).to(device)

    optimizer = Adam(ScoreNet_Parallel.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=maximum_learning_rate,
        steps_per_epoch=num_batches_per_epoch,
        epochs=n_epochs,
    )

    tqdm_epoch = tqdm.notebook.trange(n_epochs)
    losses = []
    
    for epoch in tqdm_epoch:
        # Sample a random batch
        batch_idx = random.choices(range(latent_features.shape[0]), k=batch_size)
        batch_X = latent_features[batch_idx, :]

        # Compute the loss for this batch
        loss_values = loss_fn(ScoreNet_Parallel, batch_X, T, eps)
        loss = torch.mean(loss_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Record/log the training loss
        losses.append(loss.item())
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(loss.item()))
        
    # Return the trained model. If you want a single-GPU model, you can un-wrap DataParallel:
    # ScoreNet_Parallel.module
    return ScoreNet_Parallel

def Euler_Maruyama_sampling(model, T, N, P, device):
    time_steps = torch.linspace(1., 1e-5, T) 
    step_size = time_steps[0] - time_steps[1] 

    Gen_data = torch.empty(N, P)

    init_x = torch.randn(N, P)
    X = init_x.to(device)
    
    tqdm_epoch = tqdm.notebook.trange(T)
    
    with torch.no_grad():
        for epoch in tqdm_epoch:
            time_step = time_steps[epoch].unsqueeze(0).to(device)

            # Predictor step (Euler-Maruyama)
            f = drift_coeff_fn(X, time_step).to(device)
            g = diffusion_coeff_fn(time_step).to(device)
            X = X - ( f - (g**2) * ( model(X, time_step) )  ) * step_size.to(device) + torch.sqrt(step_size).to(device)*g*torch.randn_like(X).to(device)
            tqdm_epoch.set_description('Diffusion Level: {:5f}'.format(epoch))

    Gen_data = X.cpu()
    
    return Gen_data