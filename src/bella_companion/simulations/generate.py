from phylogenie import TreeNode
from phylogenie.treesimulator import generate_trees

from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS

N_TREES = 100
MIN_TIPS = 200
MAX_TIPS = 500


def _acceptance_criterion(t: TreeNode) -> bool:
    return (
        MIN_TIPS
        <= sum(1 for leaf in t.get_leaves() if leaf.branch_length_or_raise())
        <= MAX_TIPS
    )


def generate():
    for scenario_id, scenario in SCENARIOS.items():
        generate_trees(
            output_dir=settings.simulations_data_dir / scenario_id,
            n_trees=N_TREES,
            model=scenario.model,
            max_time=scenario.max_time,
            seed=42,
            acceptance_criterion=_acceptance_criterion,
        )
