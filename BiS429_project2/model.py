"""Predictive coding network trained with inference + parameter update steps.

Each layer ``l`` predicts the layer above it as ``mu = W[l-1] f(x[l-1]) + b[l-1]``.
The prediction error ``eps[l] = (x[l] - mu) / var[l]`` drives both steps:

* :meth:`NetworkForPredictiveCoding.inference` relaxes the hidden states ``x``
  with the weights held fixed, until the free energy ``F`` stops increasing.
* :meth:`NetworkForPredictiveCoding.parameters_update` then takes one Adam step
  on the weights and biases using the settled errors.

No autograd is involved — every gradient is written out explicitly, which is
the point of the exercise. Run the whole thing under ``torch.no_grad()``.
"""

import numpy as np
import torch

import functions as F
from data import accuracy, as_tensor


class NetworkForPredictiveCoding:
    """Fully connected predictive coding network over ``cf.numperceptrons`` layers."""

    def __init__(self, cf):
        self.device = cf.device
        self.activation_function = cf.activation_function

        self.numlayers = cf.numlayers
        self.numperceptrons = cf.numperceptrons
        self.variance = cf.variance.float().to(self.device)
        self.size_of_batch = cf.size_of_batch

        # Inference hyperparameters.
        self.inference_beta_parameter = cf.inference_beta_parameter
        self.max_iterations = cf.max_iterations
        n_hidden_units = sum(cf.numperceptrons) - cf.numperceptrons[0]
        self.threshold_option = cf.threshold_option / n_hidden_units

        # Adam hyperparameters and moment estimates.
        self.type_of_optimizer = cf.type_of_optimizer
        self.lr = cf.lr
        self.adam_beta_parameter_1 = cf.adam_beta_parameter_1
        self.adam_beta_parameter_2 = cf.adam_beta_parameter_2
        self.epsilon = cf.epsilon

        self.Weight, self.bias = self._initialize_parameters()
        self.weight_constant = [torch.zeros_like(w) for w in self.Weight]
        self.bias_constant = [torch.zeros_like(b) for b in self.bias]
        self.weight_value = [torch.zeros_like(w) for w in self.Weight]
        self.bias_value = [torch.zeros_like(b) for b in self.bias]

    def _initialize_parameters(self):
        """Xavier-uniform weights and zero biases, one set per layer transition."""
        if self.activation_function != F.TANH:
            raise ValueError(f"{self.activation_function} not supported")

        weights, biases = [], []
        for l in range(self.numlayers - 1):
            fan_in, fan_out = self.numperceptrons[l], self.numperceptrons[l + 1]
            scale = np.sqrt(6 / (fan_out + fan_in))
            weight = np.random.uniform(-1, 1, size=(fan_out, fan_in)) * scale
            weights.append(as_tensor(weight, self.device))
            biases.append(as_tensor(np.zeros((fan_out, 1)), self.device))
        return weights, biases

    def _predict(self, states, layer, size_of_batch):
        """``mu`` for ``layer``: the prediction the layer below makes about it."""
        bias = self.bias[layer - 1].repeat(1, size_of_batch)
        return self.Weight[layer - 1] @ F.f(states[layer - 1], self.activation_function) + bias

    def _free_energy_and_errors(self, states, size_of_batch):
        """Return ``(F, errors)`` for the current states, both summed over layers."""
        errors = [None] * self.numlayers
        free_energy = 0.0
        for l in range(1, self.numlayers):
            mu = self._predict(states, l, size_of_batch)
            errors[l] = (states[l] - mu) / self.variance[l]
            # F is a variance-weighted sum of squared errors, not a plain sum.
            free_energy = free_energy - 0.5 * torch.sum((states[l] - mu) ** 2 / self.variance[l])
        return free_energy, errors

    def inference(self, states, size_of_batch):
        """Relax the hidden states with the parameters fixed (the E step)."""
        previous_energy, errors = self._free_energy_and_errors(states, size_of_batch)
        beta = self.inference_beta_parameter
        threshold = (self.threshold_option * self.inference_beta_parameter
                     / self.variance[self.numlayers - 1])
        stop_iteration = 0

        for iteration in range(self.max_iterations):
            # Each hidden state moves down its own error and up the error it causes above.
            for l in range(1, self.numlayers - 1):
                derivative = F.f_deriv(states[l], self.activation_function)
                states[l] = states[l] + beta * (
                    -errors[l] + derivative * (self.Weight[l].T @ errors[l + 1])
                )

            current_energy, errors = self._free_energy_and_errors(states, size_of_batch)
            energy_difference = current_energy - previous_energy

            if torch.any(energy_difference < 0):
                beta = beta / 2  # overshot: halve the step and retry
            elif torch.mean(energy_difference) < threshold:
                break

            previous_energy = current_energy
            stop_iteration = iteration

        return states, errors, stop_iteration

    def parameters_update(self, states, errors, size_of_batch, step):
        """One Adam step on the weights and biases from the settled errors (the M step)."""
        weight_gradient, bias_gradient = [], []
        for l in range(self.numlayers - 1):
            output = F.f(states[l], self.activation_function)
            weight_gradient.append(errors[l + 1] @ output.T)
            # The extra 1/size_of_batch on the bias mirrors the original submission.
            bias_gradient.append(errors[l + 1].sum(dim=1, keepdim=True) / self.size_of_batch)

            norm = self.variance[-1] / size_of_batch
            weight_gradient[l] = norm * weight_gradient[l]
            bias_gradient[l] = norm * bias_gradient[l]

        self._gradient_updates(weight_gradient, bias_gradient, step)

    def _gradient_updates(self, weight_gradient, bias_gradient, step):
        """Adam ascent on F (note the ``+``: we maximize free energy, not minimize it)."""
        if self.type_of_optimizer != "ADAM":
            raise ValueError(f"{self.type_of_optimizer} not supported")

        correction = np.sqrt(1 - self.adam_beta_parameter_2 ** step)
        beta1, beta2 = self.adam_beta_parameter_1, self.adam_beta_parameter_2

        for l in range(self.numlayers - 1):
            self.weight_constant[l] = beta1 * self.weight_constant[l] + (1 - beta1) * weight_gradient[l]
            self.bias_constant[l] = beta1 * self.bias_constant[l] + (1 - beta1) * bias_gradient[l]
            self.weight_value[l] = beta2 * self.weight_value[l] + (1 - beta2) * weight_gradient[l] ** 2
            self.bias_value[l] = beta2 * self.bias_value[l] + (1 - beta2) * bias_gradient[l] ** 2

            self.Weight[l] = self.Weight[l] + self.lr * correction * self.weight_constant[l] / (
                torch.sqrt(self.weight_value[l]) + self.epsilon
            )
            self.bias[l] = self.bias[l] + self.lr * correction * self.bias_constant[l] / (
                torch.sqrt(self.bias_value[l]) + self.epsilon
            )

    def _forward(self, x_batch, size_of_batch):
        """Feed-forward pass used to initialize the states before inference."""
        states = [None] * self.numlayers
        states[0] = x_batch
        for l in range(1, self.numlayers):
            states[l] = self._predict(states, l, size_of_batch)
        return states

    def epoch_for_train(self, x_batches, y_batches, number_epoch):
        """Run inference + parameter update over every batch of one epoch."""
        batches_number = len(x_batches)
        for batch_id, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            if batch_id % 500 == 0 and batch_id > 0:
                print(f"batch {batch_id}")

            x_batch = as_tensor(x_batch, self.device)
            y_batch = as_tensor(y_batch, self.device)
            size_of_batch = x_batch.size(1)

            states = self._forward(x_batch, size_of_batch)
            states[-1] = y_batch  # clamp the output layer to the label
            states, errors, _ = self.inference(states, size_of_batch)
            self.parameters_update(
                states, errors, size_of_batch, step=number_epoch * batches_number + batch_id
            )

    def epoch_for_test(self, x_batches, y_batches):
        """Per-batch accuracy of the plain feed-forward pass (no inference)."""
        accuracy_sets = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            x_batch = as_tensor(x_batch, self.device)
            y_batch = as_tensor(y_batch, self.device)
            states = self._forward(x_batch, x_batch.size(1))
            accuracy_sets.append(accuracy(states[-1], y_batch))
        return accuracy_sets
