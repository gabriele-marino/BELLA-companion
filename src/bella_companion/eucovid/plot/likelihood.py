from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bella_companion.backend.beast import read_log_file
from bella_companion.settings import settings


def plot_likelihood(output_file: Path):
    summaries_dir = settings.summaries_dir / "eucovid"

    glm_log = read_log_file(summaries_dir / "GLM" / "MCMC.combined.log", burn_in=0.0)
    bella_log = read_log_file(
        summaries_dir / "BELLA" / "MCMC.combined.log", burn_in=0.0
    )

    glm_values = glm_log["likelihood"]
    bella_values = bella_log["likelihood"]
    all_likelihoods = np.concatenate([glm_values, bella_values])
    bins = np.histogram_bin_edges(all_likelihoods, bins=20).tolist()

    plt.figure(layout="constrained")  # pyright: ignore
    plt.hist(  # pyright: ignore
        glm_values,
        bins=bins,
        color="C1",
        alpha=0.5,
        label="GLM",
        edgecolor="black",
        density=True,
    )
    plt.hist(  # pyright: ignore
        bella_values,
        bins=bins,
        color="C2",
        alpha=0.5,
        label="BELLA",
        edgecolor="black",
        density=True,
    )

    sns.kdeplot(glm_values.tolist(), color="C1", lw=2)
    sns.kdeplot(bella_values.tolist(), color="C2", lw=2)

    plt.xlabel("Phylogenetic likelihood")  # pyright: ignore
    plt.ylabel("Density")  # pyright: ignore
    plt.legend()  # pyright: ignore
    plt.savefig(output_file)  # pyright: ignore
