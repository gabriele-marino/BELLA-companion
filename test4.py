import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.backend.plots import ribbon_plot

CHANGE_TIMES = pd.read_csv(
    "src/bella_companion/platyrrhine/data/change_times.csv", header=None
).values.flatten()
time_bins = list(reversed([0.0, *CHANGE_TIMES, 45]))
print(len(time_bins))
log = pd.read_csv("outputs/beast/summaries/platyrrhine/MLP.csv")  # type: ignore
death_rate_estimates = log[[f"deathRateSPi{i}_3_median" for i in range(13)]]
ribbon_plot(x=time_bins, y=death_rate_estimates.values, skyline=True)
plt.gca().invert_xaxis()
plt.show()
