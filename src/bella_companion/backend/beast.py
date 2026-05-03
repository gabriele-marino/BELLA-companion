import json
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from bella_companion.backend.slurm import get_job_metadata, sbatch
from bella_companion.typings import (
    BeastDataKey,
    BeastDataValue,
    JobBatch,
    JobID,
    PosteriorWeightsByMLP,
    Weights,
)


def submit_job(
    data: Mapping[BeastDataKey, BeastDataValue],
    prefix: str | Path,
    config_path: str | Path,
    log_dir: Path,
    time: str = "240:00:00",
    cpus: int = 1,
    mem_per_cpu: int = 2000,
    seed: int = 42,
) -> JobID:
    """Submits a BEAST job to SLURM with the given data and configuration.

    Args:
        data: Dictionary of data to be passed to BEAST, where keys are variable names and values are
            file paths or string values.
        prefix: Prefix for the BEAST output files.
        config_path: Path to the BEAST XML configuration file.
        log_dir: Directory where the SLURM log files will be stored.
        time: Time limit for the SLURM job in the format "HH:MM:SS".
        cpus: Number of CPU cores to request for the SLURM job.
        mem_per_cpu: Memory per CPU in megabytes to request for the SLURM job.
        seed: Random seed to use for the BEAST run.

    Returns:
        The SLURM job ID of the submitted BEAST job.
    """
    if log_dir.exists():
        raise FileExistsError(f"Log directory {log_dir} already exists.")
    else:
        os.makedirs(log_dir, exist_ok=True)

    data_file = log_dir / ".data.tmp.json"
    with open(data_file, "w") as f:
        json.dump(data, f)
    return sbatch(
        command=" ".join(
            [
                "beast",
                f"-seed {seed}",
                f"-prefix {prefix}",
                f"-DF {data_file}",
                "-DFout /tmp/output",
                "-overwrite",
                "-statefile /tmp/state",
                str(config_path),
            ]
        )
        + f"; rm {data_file}",
        log_dir=log_dir,
        time=time,
        cpus=cpus,
        mem_per_cpu=mem_per_cpu,
    )


def read_log_file(log_file: str | Path, burn_in: int | float = 0.1) -> pd.DataFrame:
    """Reads a BEAST log file into a pandas DataFrame, applying burn-in removal.

    Args:
        log_file: Path to the BEAST log file.
        burn_in: If int, number of initial samples to discard.
            If float, fraction of samples to discard.

    Returns:
        DataFrame containing the log data after burn-in removal.
    """
    df = pd.read_csv(log_file, sep="\t", comment="#")  # pyright: ignore
    if isinstance(burn_in, float):
        chain_length: int = df["Sample"].max()
        burn_in = int(chain_length * burn_in)
    df = df[df["Sample"] > burn_in]
    df = df.drop(columns=["Sample"])
    return df


def read_weights(
    log_file: str | Path,
    burn_in: int | float = 0.1,
    n_samples: int | None = 100,
    random_seed: int | None = 42,
) -> PosteriorWeightsByMLP:
    """Reads BELLA weights from a BEAST log file.

    The weights are organized by MLP ID, with each MLP mapping to
    a list of weight samples. The MLP IDs and network architecture are
    inferred from the log file column names, which follow the pattern
    `<MLP_ID>W.Layer<layer_number>[<input_index>][<output_index>]`.

    Args:
        log_file: Path to the BEAST log file.
        burn_in: If int, number of initial samples to discard.
            If float, fraction of samples to discard.
        n_samples: Number of weight samples to return.
            If None, returns all available samples after burn-in.
        random_seed: Random seed for sampling weights when n_samples is specified.

    Returns:
        A dictionary mapping MLP IDs to lists of weight samples.
    """
    df = read_log_file(log_file, burn_in)
    if n_samples is not None:
        if n_samples > len(df):
            raise ValueError(
                "n_samples is greater than the number of available samples"
            )
        df = df.sample(n_samples, random_state=random_seed)

    targets = {
        m.group(1)
        for c in df.columns
        if (m := re.match(r"(.+?)W\.Layer\d+\[\d+\]\[\d+\]", c)) is not None
    }

    weights: dict[str, list[Weights]] = {}
    for target in targets:
        n_layers = max(
            int(re.search(r"Layer(\d+)", c).group(1))  # pyright: ignore
            for c in df.columns
            if c.startswith(f"{target}W.Layer")
        )
        n_inputs: list[int] = []
        n_outputs: list[int] = []
        for layer in range(1, n_layers + 1):
            matches = [
                re.search(r"\[(\d+)\]\[(\d+)\]", c)
                for c in df.columns
                if f"{target}W.Layer{layer}" in c
            ]
            n_inputs.append(max(int(m.group(1)) + 1 for m in matches))  # pyright: ignore
            n_outputs.append(max(int(m.group(2)) + 1 for m in matches))  # pyright: ignore

        weights[target] = [
            [
                np.array(
                    [
                        [
                            row[f"{target}W.Layer{layer + 1}[{i}][{j}]"]
                            for j in range(n_outputs[layer])
                        ]
                        for i in range(n_inputs[layer])
                    ]
                )
                for layer in range(n_layers)
            ]
            for _, row in df.iterrows()
        ]

    return weights


def summarize_log(
    log_file: str | Path,
    target_columns: list[str],
    burn_in: int | float = 0.1,
    hdi_prob: float = 0.95,
    job_id: JobID | None = None,
) -> dict[str, Any]:
    """Summarizes a BEAST log file by computing median, ESS, and HDI for target columns.

    Args:
        log_file: Path to the BEAST log file.
        target_columns: List of column names to summarize.
        burn_in: If int, number of initial samples to discard.
            If float, fraction of samples to discard.
        hdi_prob: Probability mass for the highest density interval.
        job_id: SLURM job ID for retrieving job metadata.

    Returns:
        A dictionary containing the summary statistics for each target column.
    """
    log = read_log_file(log_file, burn_in=burn_in)[target_columns]
    summary: dict[str, Any] = {"id": Path(log_file).stem, "n_samples": len(log)}
    for column in log.columns:
        summary[f"{column}.mean"] = log[column].mean()
        summary[f"{column}.std"] = log[column].std()
        summary[f"{column}.median"] = log[column].median()
        summary[f"{column}.ess"] = az.ess(np.array(log[column]))  # pyright: ignore
        lower, upper = az.hdi(np.array(log[column]), hdi_prob)  # pyright: ignore
        summary[f"{column}.lower"] = lower
        summary[f"{column}.upper"] = upper
    if job_id is not None:
        summary.update(get_job_metadata(job_id))
    return summary


def summarize_logs_dir(
    logs_dir: str | Path,
    target_columns: list[str],
    burn_in: int | float = 0.1,
    hdi_prob: float = 0.95,
    job_ids: JobBatch | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Summarizes all BEAST log files in a directory.

    Args:
        logs_dir: Directory containing BEAST log files.
        target_columns: List of column names to summarize.
        burn_in: If int, number of initial samples to discard.
            If float, fraction of samples to discard.
        hdi_prob: Probability mass for the highest density interval, by default 0.95.
        job_ids: Mapping of log file IDs to SLURM job IDs for retrieving job metadata.
        n_jobs: Number of parallel jobs to use for summarization, by default -1 (use all available cores).

    Returns:
        DataFrame containing the summary statistics for each log file.
    """
    log_files = Path(logs_dir).glob("*.log")
    summaries = Parallel(n_jobs=n_jobs)(
        delayed(
            partial(
                summarize_log,
                target_columns=target_columns,
                burn_in=burn_in,
                hdi_prob=hdi_prob,
                job_id=None if job_ids is None else job_ids[Path(log_file).stem],
            )
        )(log_file)
        for log_file in tqdm(log_files, desc="Summarizing log files")
    )
    return pd.DataFrame(summaries)


def read_weights_dir(
    logs_dir: str | Path,
    n_samples: int | None = 100,
    burn_in: int | float = 0.1,
    random_seed: int | None = 42,
    n_jobs: int = -1,
) -> list[PosteriorWeightsByMLP]:
    """Reads BELLA weights from all BEAST log files in a directory.

    Args:
        logs_dir: Directory containing BEAST log files.
        n_samples: Number of weight samples to return per log file.
            If None, returns all available samples after burn-in.
        burn_in: If int, number of initial samples to discard.
            If float, fraction of samples to discard.
        random_seed: Random seed for sampling weights when n_samples is specified, by default 42.
        n_jobs: Number of parallel jobs to use, by default -1 (use all available cores).

    Returns:
        A list of dictionaries mapping MLP IDs to their corresponding weight samples,
        one dictionary per log file.
    """
    log_files = Path(logs_dir).glob("*.log")
    return Parallel(n_jobs=n_jobs)(
        delayed(
            partial(
                read_weights,
                burn_in=burn_in,
                n_samples=n_samples,
                random_seed=random_seed,
            )
        )(log_file)
        for log_file in tqdm(log_files, desc="Reading weights from log files")
    )
