from itertools import product

import numpy as np

from bella_companion.backend import (
    Sigmoid,
    get_median_partial_dependence_plot_distribution,
    mlp_ensembles_from_logs_dir,
    ribbon_plot,
)

mlps = mlp_ensembles_from_logs_dir(
    "/Users/gmarino/bella_companion/outputs/platyrrhine",
    "deathRate",
    output_activation=Sigmoid(upper=5),
)
pdps = get_median_partial_dependence_plot_distribution(
    mlp_ensembles=mlps,
    inputs=list(
        product(np.linspace(0, 1, 10), [0, 0.3333, 0.6667, 1], np.linspace(0, 1, 10))
    ),
    feature_idx=2,
    grid=np.linspace(0, 1, 10),
)

ribbon_plot(
    x=np.linspace(0, 1, 10), y=pdps, color="blue", samples_kwargs={"linewidth": 1}
)
import matplotlib.pyplot as plt

plt.show()
