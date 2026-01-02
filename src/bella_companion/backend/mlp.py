from collections.abc import Iterator
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from bella_companion.backend.activation_functions import (
    ActivationFunctionLike,
    as_activation_function,
)
from bella_companion.backend.type_hints import Array, Weights
from bella_companion.backend.utils import read_weights


class MLP:
    def __init__(
        self,
        weights: Weights,
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ):
        """
        Initialize the MLP with given weights and activation functions.

        Parameters
        ----------
        weights : Weights
            A list of weight matrices for each layer in the MLP.
            The length of the list determines the number of layers.
            The shape of each weight matrix should be (n_inputs + 1, n_outputs),
            where n_inputs is the number of inputs to the layer (excluding bias)
            and n_outputs is the number of outputs from the layer.
        hidden_activation : ActivationFunctionLike, optional
            Activation function to use for hidden layers, by default "relu".
        output_activation : ActivationFunctionLike, optional
            Activation function to use for the output layer, by default "sigmoid".

        Raises
        ------
        ValueError
            If the output layer does not have a single output neuron.
        """
        if weights[-1].shape[1] != 1:
            raise ValueError("Output layer must have a single output neuron.")

        self._weights = weights
        self._hidden_activation = as_activation_function(activation=hidden_activation)
        self._output_activation = as_activation_function(activation=output_activation)
        self._activations = [self._hidden_activation] * (self.n_layers - 1) + [
            self._output_activation
        ]

    @property
    def n_layers(self) -> int:
        return len(self._weights)

    def forward(self, inputs: ArrayLike) -> Array:
        """
        Perform a forward pass through the MLP.

        Parameters
        ----------
        inputs : ArrayLike
            Input data of shape (n_samples, n_features).

        Returns
        -------
        Array
            The output of the MLP after the forward pass, of shape (n_samples,).
        """
        x = np.asarray(inputs, dtype=np.float64)
        n_samples, _ = x.shape
        for layer_weights, activation in zip(self._weights, self._activations):
            bias = np.ones((n_samples, 1))
            x = np.hstack((bias, x))
            x = np.dot(x, layer_weights)
            x = activation(x)
        return x.flatten()

    def __call__(self, inputs: ArrayLike) -> Array:
        return self.forward(inputs)

    def __repr__(self) -> str:
        return (
            f"MLP(n_layers={self.n_layers}, "
            f"hidden_activation={self._hidden_activation}, "
            f"output_activation={self._output_activation})"
        )


class MLPEnsemble:
    def __init__(
        self,
        weights_list: list[Weights],
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ):
        """
        Initialize the MLP ensemble with given weights and activation functions.

        Parameters
        ----------
        weights_list : list[Weights]
            A list of weight sets, where each set corresponds to an MLP in the ensemble.
        hidden_activation : ActivationFunctionLike, optional
            Activation function to use for hidden layers, by default "relu".
        output_activation : ActivationFunctionLike, optional
            Activation function to use for the output layer, by default "sigmoid".
        """
        self._mlps = [
            MLP(
                weights=weights,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
            )
            for weights in weights_list
        ]

    @property
    def n_models(self) -> int:
        return len(self._mlps)

    @property
    def mlps(self) -> tuple[MLP, ...]:
        return tuple(self._mlps)

    def forward(
        self,
        inputs: ArrayLike,  # (n_samples, n_features)
    ) -> Array:  # (n_models, n_samples)
        """
        Perform a forward pass through the MLP ensemble.

        Parameters
        ----------
        inputs : ArrayLike
            Input data of shape (n_samples, n_features).

        Returns
        -------
        Array
            The outputs of the MLP ensemble after the forward pass,
            of shape (n_models, n_samples).
        """
        return np.array([mlp(inputs) for mlp in self._mlps])

    def forward_median(
        self,
        inputs: ArrayLike,  # (n_samples, n_features)
    ) -> Array:  # (n_samples,)
        """
        Perform a forward pass through the MLP ensemble and compute the median output.

        Parameters
        ----------
        inputs : ArrayLike
            Input data of shape (n_samples, n_features).

        Returns
        -------
        Array
            The median output of the MLP ensemble after the forward pass,
            of shape (n_samples,).
        """
        outputs = self.forward(inputs)  # (n_models, n_samples)
        return np.median(outputs, axis=0)  # (n_samples,)

    @classmethod
    def from_log_file(
        cls,
        log_file: str | Path,
        target_name: str,
        burn_in: int | float = 0.1,
        n_samples: int | None = 100,
        random_seed: int | None = 42,
        hidden_activation: ActivationFunctionLike = "relu",
        output_activation: ActivationFunctionLike = "sigmoid",
    ) -> "MLPEnsemble":
        """
        Create an MLPEnsemble instance from a BEAST log file.

        Parameters
        ----------
        log_file : str | Path
            Path to the BEAST log file.
        target_name : str
            The target variable name for which to extract weights.
        burn_in : int | float, optional
            If int, number of initial samples to discard.
            If float, fraction of samples to discard, by default 0.1.
        n_samples : int | None, optional
            Number of weight samples to return, by default 100.
            If None, returns all available samples after burn-in.
        random_seed : int | None, optional
            Random seed for sampling weights when n_samples is specified, by default 42.
        hidden_activation : ActivationFunctionLike, optional
            Activation function to use for hidden layers, by default "relu".
        output_activation : ActivationFunctionLike, optional
            Activation function to use for the output layer, by default "sigmoid".

        Returns
        -------
        MLPEnsemble
            An instance of MLPEnsemble initialized with weights from the log file.
        """
        weights_list = read_weights(
            log_file=log_file,
            burn_in=burn_in,
            n_samples=n_samples,
            random_seed=random_seed,
        )
        return cls(
            weights_list=weights_list[target_name],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

    def __call__(self, inputs: ArrayLike) -> Array:
        return self.forward(inputs)

    def __len__(self) -> int:
        return self.n_models

    def __iter__(self) -> Iterator[MLP]:
        return iter(self._mlps)

    def __repr__(self) -> str:
        return f"MLPEnsemble(n_models={self.n_models})"


def mlp_ensembles_from_logs_dir(
    logs_dir: str | Path,
    target_name: str,
    burn_in: int | float = 0.1,
    n_samples: int | None = 100,
    random_seed: int | None = 42,
    hidden_activation: ActivationFunctionLike = "relu",
    output_activation: ActivationFunctionLike = "sigmoid",
    n_jobs: int = -1,
) -> list[MLPEnsemble]:
    """
    Create a list of MLPEnsemble instances from all BEAST log files in a directory.

    Parameters
    ----------
    logs_dir : str | Path
        Path to the directory containing BEAST log files.
    target_name : str
        The target variable name for which to extract weights.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.
    n_samples : int | None, optional
        Number of weight samples to return, by default 100.
        If None, returns all available samples after burn-in.
    random_seed : int | None, optional
        Random seed for sampling weights when n_samples is specified, by default 42.
    hidden_activation : ActivationFunctionLike, optional
        Activation function to use for hidden layers, by default "relu".
    output_activation : ActivationFunctionLike, optional
        Activation function to use for the output layer, by default "sigmoid".
    n_jobs : int, optional
        Number of parallel jobs to use, by default -1 (use all available cores).

    Returns
    -------
    list[MLPEnsemble]
        A list of MLPEnsemble instances initialized with weights from the log files.
    """
    log_files = list(Path(logs_dir).glob("*.log"))
    return Parallel(n_jobs=n_jobs)(
        delayed(MLPEnsemble.from_log_file)(
            log_file=log_file,
            target_name=target_name,
            burn_in=burn_in,
            n_samples=n_samples,
            random_seed=random_seed,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        for log_file in tqdm(log_files, desc="Loading MLP ensembles from logs")
    )
