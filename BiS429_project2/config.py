"""Default configuration for the predictive coding network.

``cf`` is an attribute-accessible dict so the model can read ``cf.lr`` etc.
:func:`default_config` returns the settings that produced the run recorded in
``results/train_log.txt``.
"""

import torch

import functions as F


class AttrDict(dict):
    """Dict whose keys are also reachable as attributes."""

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def default_config():
    cf = AttrDict()

    # Training loop.
    cf.n_epochs = 100
    cf.size_of_batch = 128
    cf.data_size = None  # None uses the full 60k/10k split
    cf.seed = None

    # Preprocessing.
    cf.apply_inv = True
    cf.apply_scaling = True
    cf.label_scale = 0.94
    cf.img_scale = 1.0

    # Architecture.
    cf.numperceptrons = [784, 500, 500, 10]
    cf.activation_function = F.TANH
    cf.numlayers = len(cf.numperceptrons)
    cf.var_out = 1
    cf.variance = torch.ones(cf.numlayers)

    # Inference (E) step.
    cf.inference_beta_parameter = 0.1
    cf.max_iterations = 50
    cf.threshold_option = 1e-6

    # Parameter update (M) step.
    cf.type_of_optimizer = "ADAM"
    cf.lr = 1e-3
    cf.adam_beta_parameter_1 = 0.9
    cf.adam_beta_parameter_2 = 0.999
    cf.epsilon = 1e-8
    cf.decay_r = 0.9

    cf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return cf
