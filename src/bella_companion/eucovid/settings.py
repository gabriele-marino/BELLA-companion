from pathlib import Path

import numpy as np

from bella_companion.settings import settings
from bella_companion.typings import Array

BEAST_CONFIGS_DIR = Path(__file__).parent / "beast_configs"

N_SEEDS = 3

COLORS = {
    "China": "#F0E442",
    "France": "#009E73",
    "Germany": "#D55E00",
    "Italy": "#E69F00",
    "OtherEU": "#56B4E9",
}
COUNTRIES = list(COLORS.keys())
N_COUNTRIES = len(COUNTRIES)

N_CASES = [80814, 716, 847, 5883, 3054]
"""Number of confirmed cases in each country before 2020-03-08, derived from ECDC data."""

DATA_DIR = Path(__file__).parent / "data"
MSA_FILE = DATA_DIR / "msa.fasta"


def _transform(x: Array) -> Array:
    x_log = np.log(x + 1)
    return (x_log - x_log.mean()) / x_log.std(ddof=1)


PREDICTOR_FILE = DATA_DIR / "predictor.tsv"
PREDICTOR = np.loadtxt(PREDICTOR_FILE)
"""Matrix with N_TIME_BINS * N_COUNTRIES rows and N_COUNTRIES - 1 columns, containing the number of daily air passengers per 100,000 inhabitants traveling from each country to every other country within each time bin, ordered from most recent to least recent. Data are derived from the Eurostat datasets AVIA_PAOCC and AVIA_PAEXCC. Credits: Cecilia Valenzuela Agüí."""
PREDICTOR_TRANSFORMED = _transform(PREDICTOR)

BELLA_CONFIGS = settings.bella_model_configs["BELLA-3"]
