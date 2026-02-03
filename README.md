# BELLA-Companion

### ⚠️🚧🚨 The documentation for this repository is still under development.

## CLI

The CLI entrypoint is `bella` (see `src/bella_companion/cli.py`). It requires an `.env` file to be present in the current working directory defining the settings for the analyses to be run.

### Usage

```bash
bella <command> <subcommand> <...>
```

### Commands

```text
sim
  generate            Generate synthetic simulation datasets.
  run                 Run BEAST2 analyses on simulated datasets.
  summarize           Summarize BEAST2 log outputs for simulated datasets.
  metrics             Compute and print metrics for simulated datasets.
  plot
    all               Generate plots for all simulation scenarios.
    epi-multitype     Generate plots for the epi-multitype scenario.
    epi-skyline       Generate plots for the epi-skyline scenarios.
    fbd-2traits       Generate plots for the fbd-2traits scenario.
    fbd-no-traits     Generate plots for the fbd-no-traits scenarios.
    scenarios         Generate scenario overview plots.

platyrrhine
  run                 Run BEAST2 analyses on empirical platyrrhine datasets.
  summarize           Summarize BEAST2 log outputs for empirical datasets.
  plot
    all               Generate plots for all platyrrhine datasets.
    estimates         Generate parameter estimate plots.
    trees             Generate tree-mapped parameter estimate plots.
    shap              Generate SHAP plots.

eucovid
  run                 Run BEAST2 analyses on empirical eucovid datasets.
  summarize           Summarize BEAST2 log outputs for empirical datasets.
  plot
    all                       Generate plots for all eucovid datasets.
    likelihood                Generate likelihood distribution plots.
    sankey                    Generate sankey plots.
    trees                     Generate tree plots.
    flights-and-populations   Plots for the flights and populations scenario.
    flights-over-populations  Plots for the flights over populations scenario.
```