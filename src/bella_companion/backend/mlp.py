from pathlib import Path

import numpy as np
from jaxtyping import Float

from bella_companion.backend.activation_functions import (
    ActivationFunctionLike,
    as_activation_function,
)
from bella_companion.backend.beast import read_weights
from bella_companion.typings import Array, PosteriorWeights, Weights


class MLP:
    """Multilayer Perceptron (MLP) implementation for forward passes with given weights and activation functions.

    Args:
        weights: Weight matrices for each layer in the MLP.
            The length of the sequence determines the number of layers.
            The shape of each weight matrix should be (n_nodes_in + 1, n_nodes_out),
            where n_nodes_in is the number of input nodes (excluding bias)
            and n_nodes_out is the number of output nodes for that layer.
            The last layer must have a single output node.
        hidden_activation: Activation function to use for hidden layers.
        output_activation: Activation function to use for the output layer.
    """

    def __init__(
        self,
        weights: Weights,
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ):
        """Initialize the MLP with given weights and activation functions."""

        if weights[-1].shape[1] != 1:
            raise ValueError("Output layer must have a single output neuron.")

        self._weights = weights
        hidden_activation = as_activation_function(hidden_activation)
        output_activation = as_activation_function(output_activation)

        n_layers = len(weights)
        self._activations = [hidden_activation] * (n_layers - 1) + [output_activation]

    def __call__(
        self,
        x: Float[Array, "batch_size n_features"],  # noqa: F722
    ) -> Float[Array, "batch_size"]:  # noqa: F821
        """Perform a forward pass through the MLP."""
        batch_size, _ = x.shape
        for layer_weights, activation in zip(self._weights, self._activations):
            bias = np.ones((batch_size, 1))
            x = np.hstack((bias, x))
            x = np.dot(x, layer_weights)
            x = activation(x)
        return x.flatten()


class BayesMLP:
    """A wrapper around MLP that allows for multiple sets of weights, representing samples from a posterior distribution.

    Args:
        posterior_weights: A sequence of weight sets, where each set corresponds to a sample from the posterior distribution.
        hidden_activation: Activation function to use for hidden layers.
        output_activation: Activation function to use for the output layer.
    """

    def __init__(
        self,
        posterior_weights: PosteriorWeights,
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ):
        """Initialize the BayesMLP with given posterior weights and activation functions."""
        self._mlps = [
            MLP(
                weights=weights,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
            )
            for weights in posterior_weights
        ]

    def __call__(
        self,
        inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    ) -> Float[Array, "n_samples batch_size"]:  # noqa: F722
        """Perform a forward pass through each sampled MLP and return the outputs."""
        return np.array([mlp(inputs) for mlp in self._mlps])

    @classmethod
    def from_log_file(
        cls,
        log_file: Path,
        id: str,
        burn_in: int | float = 0.1,
        n_samples: int | None = 100,
        random_seed: int | None = 42,
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ) -> "BayesMLP":
        """Load weights from a BEAST log file and create a BayesMLP instance.

        Args:
            log_file: Path to the BEAST log file.
            id: The identifier for the model for which to extract weights.
            burn_in: If int, number of initial samples to discard.
                If float, fraction of samples to discard.
            n_samples: Number of samples to return.
                If None, returns all available samples after burn-in.
            random_seed: Random seed for sampling weights when n_samples is specified.
            hidden_activation: Activation function to use for hidden layers.
            output_activation: Activation function to use for the output layer.
        """
        posterior_weights = read_weights(
            log_file=log_file,
            burn_in=burn_in,
            n_samples=n_samples,
            random_seed=random_seed,
        )
        return cls(
            posterior_weights=posterior_weights[id],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

    def __iter__(self):
        """Allow iteration over the sampled MLPs."""
        return iter(self._mlps)

    def __len__(self):
        """Return the number of sampled MLPs."""
        return len(self._mlps)
