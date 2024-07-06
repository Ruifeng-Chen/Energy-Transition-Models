import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets import MLP


class MLPETM(nn.Module):

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dims,
            activation,
            with_reward: bool = False,
            device: str = "cpu",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.activation = activation
        self.with_reward = with_reward

        self.mlp = MLP(2 * obs_dim + action_dim + with_reward, hidden_dims, 1, dropout_rate=None)

        self.to(self.device)

    def forward(self, state_action, delta_state):
        state_action = torch.as_tensor(state_action, dtype=torch.float32).to(self.device)

        delta_state = torch.as_tensor(delta_state, dtype=torch.float32).to(self.device)

        input = torch.cat([state_action, delta_state], -1)

        output = self.mlp(input)
        return output.squeeze(axis=-1)
    
def grad_wrt_next_s(
        energy_network,
        state_action,
        delta_state,
        create_graph: bool = False,
):
    delta_state.requires_grad = True
    energies = energy_network(state_action, delta_state)
    grad = torch.autograd.grad(energies.sum(), delta_state, create_graph=create_graph)[0]
    return grad, energies

def langevin_step(
    energy_network,
    state_action,
    delta_state,
    noise_scale,
    grad_clip,
    delta_clip,
    margin_clip,
    stepsize,

):
    l_lambda = 1.0
    grad, energy = grad_wrt_next_s(energy_network, state_action, delta_state)

    grad_norm = torch.norm(grad, dim=-1, keepdim=True)

    if grad_clip:
        grad = torch.clamp(grad, -grad_clip, grad_clip)

    delta_state_drift = stepsize * (0.5 * l_lambda * grad + torch.randn(delta_state.shape, device=grad.device) * l_lambda * noise_scale)

    if delta_clip:
       delta_state_drift = torch.clamp(delta_state_drift, -delta_clip, delta_clip)

    delta_state = delta_state - delta_state_drift

    if margin_clip is not None:
        delta_state = torch.clamp(delta_state, -margin_clip, margin_clip)
    return delta_state, energy, grad_norm

class ExponentialSchedule:
  """Exponential learning rate schedule for Langevin sampler."""

  def __init__(self, init, decay):
    self._decay = decay
    self._latest_lr = init

  def get_rate(self, index):
    """Get learning rate. Assumes calling sequentially."""
    del index
    self._latest_lr *= self._decay
    return self._latest_lr
  
class PolynomialSchedule:
  """Polynomial learning rate schedule for Langevin sampler."""

  def __init__(self, init, final, power, num_steps):
    self._init = init
    self._final = final
    self._power = power
    self._num_steps = num_steps

  def get_rate(self, index):
    """Get learning rate for index."""
    return ((self._init - self._final) *
            ((1 - (float(index) / float(self._num_steps-1))) ** (self._power))
            ) + self._final

def langevin_mcmc_sa_s(
        energy_network,
        state_action,
        delta_state,
        num_iterations=25,
        sampler_stepsize_init=1e-1,
        sampler_stepsize_decay=0.8,
        noise_scale=1.0,
        grad_clip=None,
        delta_clip=None,
        margin_clip=None,
        use_polynomial_rate=True,  # default is exponential
        sampler_stepsize_final=1e-5,  # if using polynomial langevin rate.
        sampler_stepsize_power=2.0,  # if using polynomial langevin rate.
):
    stepsize = sampler_stepsize_init

    if use_polynomial_rate:
        schedule = PolynomialSchedule(sampler_stepsize_init, sampler_stepsize_final,
                                    sampler_stepsize_power, num_iterations)
    else:  # default to exponential rate
        schedule = ExponentialSchedule(sampler_stepsize_init,
                                    sampler_stepsize_decay)

    for step in range(num_iterations):

        delta_state, _, _ = langevin_step(energy_network, 
                                                      state_action,
                                                      delta_state,
                                                      noise_scale,
                                                      grad_clip,
                                                      delta_clip,
                                                      margin_clip,
                                                      stepsize)
        delta_state = delta_state.detach()
        stepsize = schedule.get_rate(step + 1)


    return delta_state