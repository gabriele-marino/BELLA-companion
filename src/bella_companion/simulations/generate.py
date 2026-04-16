import os
from pathlib import Path

from phylogenie import TreeNode
from phylogenie.treesimulator import generate_trees

from bella_companion.simulations.scenarios import SCENARIOS

N_TREES = 100
MIN_TIPS = 200
MAX_TIPS = 500


def _acceptance_criterion(t: TreeNode) -> bool:
    return (
        MIN_TIPS
        <= sum(1 for leaf in t.get_leaves() if leaf.branch_length > 0)  # pyright: ignore
        <= MAX_TIPS
    )


def generate():
    base_output_dir = Path(os.environ["BELLA_SIMULATIONS_DATA_DIR"])
    for scenario_name, scenario in SCENARIOS.items():
        generate_trees(
            output_dir=base_output_dir / scenario_name,
            n_trees=N_TREES,
            model=scenario.model,
            max_time=scenario.max_time,
            seed=42,
            n_jobs=1,
            acceptance_criterion=_acceptance_criterion,
        )
