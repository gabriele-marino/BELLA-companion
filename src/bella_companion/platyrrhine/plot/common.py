import matplotlib.pyplot as plt  # pyright: ignore
import numpy as np

_GRADIENT = np.linspace(0.4, 0.9, 4)
COLORS: dict[str, np.typing.NDArray[np.floating]] = {
    "birth": plt.cm.Blues(_GRADIENT),  # pyright: ignore
    "death": plt.cm.Oranges(_GRADIENT),  # pyright: ignore
    "diversification": plt.cm.Greens(_GRADIENT),  # pyright: ignore
}

RATE_LABELS = {
    "birth": r"$\lambda$",
    "death": r"$\mu$",
    "diversification": r"$d$",
}
TYPE_LABELS = {0: "0 (Tiny)", 1: "1 (Small)", 2: "2 (Medium)", 3: "3 (Large)"}
