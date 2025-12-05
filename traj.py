import pandas as pd

change_times = [0.019125683060110532, 0.0983606557379062, 0.18306010928972682]

traj = pd.concat(
    [
        pd.read_csv(
            f"outputs/beast/runs/eucovid/flights_and_populations/BELLA/{seed}/trajectories.csv",
            sep="\t",
        ).assign(seed=seed)
        for seed in range(1, 4)
    ],
    ignore_index=True,
)

traj = traj[traj["Sample"] >= 1_000_000]
print(
    traj.groupby(["Sample", "seed"])
    .apply(
        lambda g: g.query(" type == 4 & variable == 'M'")["value"].sum(),
        include_groups=False,
    )
    .median()
)
