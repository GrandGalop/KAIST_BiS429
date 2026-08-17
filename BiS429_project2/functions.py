"""Activation functions, their derivatives and inverses.

``f`` and ``f_deriv`` operate on torch tensors shaped ``(activation_size,
batch_size)``. ``f_inv`` is applied to the raw numpy images during
preprocessing, before they are moved onto the device.
"""

import numpy as np
import torch

LINEAR = "LINEAR"
TANH = "TANH"
LOGSIG = "LOGSIG"

EPS = 1e-7


def _logsig(x):
    return 1.0 / (1.0 + torch.exp(-x))


def f(x, act_fn):
    """Apply the activation function."""
    if act_fn == LINEAR:
        return x
    if act_fn == TANH:
        return torch.tanh(x)
    if act_fn == LOGSIG:
        return _logsig(x)
    raise ValueError(f"{act_fn} not supported")


def f_deriv(x, act_fn):
    """Derivative of the activation function, evaluated elementwise at ``x``."""
    if act_fn == LINEAR:
        return torch.ones_like(x)
    if act_fn == TANH:
        return 1.0 - torch.tanh(x) ** 2
    if act_fn == LOGSIG:
        sigma = _logsig(x)
        return sigma * (1.0 - sigma)
    raise ValueError(f"{act_fn} not supported")


def f_inv(x, act_fn):
    """Inverse of the activation function, used to un-squash the input images."""
    if act_fn == LINEAR:
        return x
    if act_fn == TANH:
        return 0.5 * np.log((1.0 + x) / (1.0 - x + EPS))
    if act_fn == LOGSIG:
        return np.log(x / (1.0 - x + EPS) + EPS)
    raise ValueError(f"{act_fn} not supported")
