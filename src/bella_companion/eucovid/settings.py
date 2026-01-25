from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MSA_FILE = DATA_DIR / "msa.fasta"

COLORS = {
    "China": "#F0E442",
    "France": "#009E73",
    "Germany": "#D55E00",
    "Italy": "#E69F00",
    "OtherEU": "#56B4E9",
}
COUNTRIES = list(COLORS.keys())
N_COUNTRIES = len(COUNTRIES)
