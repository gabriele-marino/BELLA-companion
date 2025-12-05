from itertools import product

import numpy as np

from bella_companion.backend import (
    Sigmoid,
    get_median_shap_feature_importance_distribution,
    mlp_ensembles_from_logs_dir,
)

mlps = mlp_ensembles_from_logs_dir(
    "/Users/gmarino/bella_companion/outputs/platyrrhine",
    "deathRate",
    output_activation=Sigmoid(upper=5),
)
median_shap_importances = get_median_shap_feature_importance_distribution(
    ensembles=mlps,
    inputs=list(
        product(np.linspace(0, 1, 10), [0, 0.3333, 0.6667, 1], np.linspace(0, 1, 10))
    ),
)

import joblib

joblib.dump(median_shap_importances, "shape_importances.pkl")
