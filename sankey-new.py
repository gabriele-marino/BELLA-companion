import numpy as np
import pandas as pd
import plotly.graph_objects as go

from bella_companion.backend import read_log_file

ages = [0.18306010928972682, 0.0993606557379062, 0.019125683060110532, 0]

traj = pd.concat(
    [
        pd.read_csv(
            f"outputs/runs/eucovid/flights_over_population/GLM/{seed}/trajectories.csv",
            sep="\t",
        ).assign(seed=seed)
        for seed in range(1, 4)
    ],
    ignore_index=True,
)

traj = traj[traj["Sample"] >= 1_000_000]
COUNTRIES = ["China", "France", "Germany", "Italy", "OtherEU"]


def _get_migration_rate(i: int, j: int, t: int) -> float:
    t_min, t_max = ages[t + 1], ages[t]

    def compute_group(g):
        return g.query(
            "(@t_max > age) and (age > @t_min) and variable == 'M' and type == @i and type2 == @j",
            local_dict={"t_max": t_max, "t_min": t_min, "i": i, "j": j},
        )["value"].sum()

    mig_rate = (
        traj.groupby(["Sample", "seed"])
        .apply(compute_group, include_groups=False)
        .median()
    )
    print(
        f"Migration rate {COUNTRIES[i]} -> {COUNTRIES[j]} at time bin {t}: {mig_rate}"
    )
    if mig_rate == 0.0:
        mig_rate = 1e-10
    return mig_rate


N_TIME_BINS = 3
N_LAYERS = N_TIME_BINS + 1
N_COUNTRIES = len(COUNTRIES)
COLORS = ["#F0E442", "#009E73", "#D55E00", "#E69F00", "#56B4E9"]
MIGRATIONS = [
    _get_migration_rate(i, j, t)
    for t in range(N_TIME_BINS)
    for i, source in enumerate(COUNTRIES)
    for j, target in enumerate(COUNTRIES)
    if target != source
]
# print(MIGRATIONS)
"""
MIGRATIONS = [
    np.float64(4.0),
    np.float64(4.0),
    np.float64(3.0),
    np.float64(5.0),
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    np.float64(1.0),
    1e-10,
    1e-10,
    1e-10,
    1e-10,
    np.float64(3.0),
    np.float64(4.0),
    np.float64(1.0),
    np.float64(5.0),
    np.float64(1.0),
    np.float64(5.0),
    np.float64(6.0),
    np.float64(13.0),
    1e-10,
    np.float64(1.0),
    np.float64(2.0),
    np.float64(6.0),
    np.float64(1.0),
    np.float64(22.0),
    np.float64(21.0),
    np.float64(50.0),
    np.float64(1.0),
    np.float64(7.0),
    np.float64(9.0),
    np.float64(7.0),
    np.float64(1.0),
    np.float64(1.0),
    1e-10,
    1e-10,
    np.float64(3.0),
    np.float64(11.0),
    np.float64(7.0),
    np.float64(29.0),
    1e-10,
    np.float64(1.0),
    np.float64(1.0),
    np.float64(5.0),
    np.float64(1.0),
    np.float64(13.0),
    np.float64(15.0),
    np.float64(34.0),
    np.float64(2.0),
    np.float64(9.0),
    np.float64(12.0),
    np.float64(5.0),
]
"""
source = np.repeat(list(range(N_COUNTRIES * N_TIME_BINS)), N_COUNTRIES - 1)
target = [
    x + N_COUNTRIES * t
    for t in range(1, N_TIME_BINS + 1)
    for i in list(range(N_COUNTRIES))
    for x in list(range(N_COUNTRIES))
    if x != i
]

x = np.repeat([0.1, 0.4, 0.7, 1], N_COUNTRIES)
y = np.repeat([0.56, 0.32, 0.175, 0.3], N_COUNTRIES)
y[9] = 0.2
y[5] = 0.62
y[18] = 0.22
y[17] = 0.35
fig = go.Figure(
    go.Sankey(  # pyright: ignore
        arrangement="snap",
        node=dict(color=COLORS * (N_TIME_BINS + 1), x=x, y=y),
        link=dict(source=source, target=target, value=MIGRATIONS),
    )
)

fig.update_layout()
fig.write_image("sankey.svg")
