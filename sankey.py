import numpy as np
import plotly.graph_objects as go

from bella_companion.backend import read_log_file

log = read_log_file(
    "outputs/beast/runs/eucovid/flights_over_population/BELLA/CombinedMCMC.log"
)

POPULATIONS = {
    "China": 1411.7,
    "France": 67.4,
    "Germany": 83.2,
    "Italy": 59.6,
    "OtherEU": 237.5,
}

N_TIME_BINS = 3
N_LAYERS = N_TIME_BINS + 1
COUNTRIES = ["OtherEU", "France", "Italy", "Germany", "China"]
N_COUNTRIES = len(COUNTRIES)
COLORS = ["#56B4E9", "#009E73", "#E69F00", "#D55E00", "#F0E442"]
MIGRATIONS = [
    np.median(log[f"migrationRateSPi{i}_{source}_to_{target}"]) * POPULATIONS[source]
    for i in range(1, 4)
    for source in COUNTRIES
    for target in COUNTRIES
    if target != source
]

source = np.repeat(list(range(N_COUNTRIES * N_TIME_BINS)), N_COUNTRIES - 1)
target = [
    x + N_COUNTRIES * t
    for t in range(1, N_TIME_BINS + 1)
    for i in list(range(N_COUNTRIES))
    for x in list(range(N_COUNTRIES))
    if x != i
]

x = np.repeat([0.1, 0.4, 0.7, 1], N_COUNTRIES)
y = np.repeat([0.185, 0.135, 0.175, 0.31], N_COUNTRIES)
y[-3] = 0.29  # Snap adjustment for visual clarity
fig = go.Figure(
    go.Sankey(  # pyright: ignore
        arrangement="snap",
        node=dict(color=COLORS * (N_TIME_BINS + 1), x=x, y=y),
        link=dict(source=source, target=target, value=MIGRATIONS),
    )
)

fig.update_layout()
fig.write_image("sankey.svg")
