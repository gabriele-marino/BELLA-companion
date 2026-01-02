import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests
from phylogenie import Sequence, dump_fasta, load_fasta

from bella_companion.eucovid.settings import DATA_DIR, MSA_FILE

LAST_TIME_BIN = "2020-03-08"
TIME_BINS_ISOFORMAT = ["2019-12-31", "2020-01-31", "2020-02-29", LAST_TIME_BIN]
TIME_BINS = list(map(date.fromisoformat, TIME_BINS_ISOFORMAT))
N_TIME_BINS = len(TIME_BINS)
CORE_COUNTRIES = ["Italy", "France", "Germany"]
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

LAPIS_URL = "https://lapis.cov-spectrum.org/open/v2/sample/alignedNucleotideSequences"


def _download_full_msa(output_file: str | Path) -> None:
    params = {
        "dateTo": LAST_TIME_BIN,
        "host": "Homo sapiens",
        "country": ["China", *CORE_COUNTRIES, *OTHER_EUROPEAN_COUNTRIES],
        "downloadAsFile": "true",
        "fastaHeaderTemplate": "{genbankAccessionRev}|{country}|{date}",
    }

    response = requests.get(LAPIS_URL, params=params)
    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)


def download_data():
    print("Downloading full MSA...")
    full_msa_filepath = DATA_DIR / ".full_msa.tmp.fasta"
    _download_full_msa(full_msa_filepath)
    full_msa = load_fasta(full_msa_filepath)
    os.remove(full_msa_filepath)

    print("Filtering sequences...")
    filtered_sequences: dict[str, list[list[Sequence]]] = defaultdict(
        lambda: [[] for _ in range(N_TIME_BINS)]
    )
    for seq in full_msa:
        _, country, dt = seq.id.split("|")
        time_bin = next(
            i for i, tb in enumerate(TIME_BINS) if tb >= date.fromisoformat(dt)
        )
        filtered_sequences[country][time_bin].append(seq)
    sequences = [
        seq
        for country in ["Germany", "Italy", "France"]
        for time_bin in filtered_sequences[country]
        for seq in time_bin
    ]
    sequences += [
        seq
        for country in OTHER_EUROPEAN_COUNTRIES + ["China"]
        for time_bin in filtered_sequences[country]
        for seq in time_bin[:MAX_SEQUENCES_PER_COUNTRY_PER_TIME_BIN]
    ]

    dump_fasta(sequences, MSA_FILE)
    print(f"Filtered MSA saved to {MSA_FILE}")
