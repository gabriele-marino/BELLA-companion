import numpy as np

from bella_companion.backend import Sigmoid, mlp_ensembles_from_logs_dir, ribbon_plot

mlps = mlp_ensembles_from_logs_dir(
    "outputs/beast/runs/platyrrhine",
    target_name="birthRate",
    output_activation=Sigmoid(upper=5),
)
estimates = [
    mlp.forward_median([[i, 0] for i in np.linspace(0, 1, 10)]) for mlp in mlps
]
ribbon_plot(estimates, color="C0", skyline=True)
