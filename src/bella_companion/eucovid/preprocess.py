from collections import defaultdict
from datetime import date

from phylogenie import Sequence, dump_fasta, load_fasta

from bella_companion.eucovid.settings import DATA_DIR, MSA_FILE

TIME_BINS = list(
    map(date.fromisoformat, ["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-09"])
)
N_TIME_BINS = len(TIME_BINS)
OTHER_EUROPEAN_COUNTRIES = [
    "Greece",
    "United Kingdom",
    "Finland",
    "Poland",
    "Sweden",
    "Belgium",
    "Netherlands",
    "Spain",
    "Iceland",
    "Switzerland",
]
MAX_SEQUENCES_PER_COUNTRY_PER_TIME_BIN = 10


def main():
    msa = load_fasta(DATA_DIR / "all.fasta")

    seqs: dict[str, list[list[Sequence]]] = defaultdict(
        lambda: [[] for _ in range(N_TIME_BINS)]
    )
    for seq in msa:
        _, country, dt = seq.id.split("|")
        time_bin = next(
            i for i, tb in enumerate(TIME_BINS) if tb > date.fromisoformat(dt)
        )
        seqs[country][time_bin].append(seq)
    sequences = [
        seq
        for country in ["Germany", "Italy", "France"]
        for time_bin in seqs[country]
        for seq in time_bin
    ]
    sequences += [
        seq
        for country in OTHER_EUROPEAN_COUNTRIES + ["China"]
        for time_bin in seqs[country]
        for seq in time_bin[:MAX_SEQUENCES_PER_COUNTRY_PER_TIME_BIN]
    ]

    dump_fasta(sequences, MSA_FILE)


if __name__ == "__main__":
    main()
