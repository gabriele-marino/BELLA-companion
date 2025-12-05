from itertools import product

import joblib
import numpy as np
import shap
from tqdm import tqdm

from bella_companion.backend import MLPEnsemble, Sigmoid, Weights

w: list[dict[str, list[Weights]]] = joblib.load(
    "outputs/beast/summaries/platyrrhine/MLP.weights.pkl"
)
print(len(w[0]["deathRate"]))
mlps = MLPEnsemble(w[0]["deathRate"], "relu", Sigmoid(upper=5))
inputs = list(product(np.linspace(0, 1, 10).tolist(), [0, 1, 2, 3]))

inputs = np.asarray(inputs, dtype=np.float64)
explained = [shap.Explainer(mlp, inputs)(inputs) for mlp in tqdm(mlps)]
