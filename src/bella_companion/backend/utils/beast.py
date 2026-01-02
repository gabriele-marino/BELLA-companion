import re
from functools import partial
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from bella_companion.backend.type_hints import Weights
from bella_companion.backend.utils.slurm import get_job_metadata

MEDIAN_POSTFIX = "_median"
LOWER_POSTFIX = "_lower"
UPPER_POSTFIX = "_upper"
ESS_POSTFIX = "_ess"


def read_log_file(log_file: str | Path, burn_in: int | float = 0.1) -> pd.DataFrame:
    """
    Reads a BEAST log file into a pandas DataFrame, applying burn-in removal.

    Parameters
    ----------
    log_file : str | Path
        Path to the BEAST log file.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.

    Returns
    -------
    pd.DataFrame
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
) -> dict[str, list[Weights]]:
    """
    Reads BELLA weights from a BEAST log file.

    The weights are organized by target name, with each target mapping to
    a list of weight samples. The target names and network architecture are
    inferred from the log file column names, which follow the pattern
    `<target_name>W.Layer<layer_number>[<input_index>][<output_index>]`.

    Parameters
    ----------
    log_file : str | Path
        Path to the BEAST log file.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.
    n_samples : int | None, optional
        Number of weight samples to return, by default 100.
        If None, returns all available samples after burn-in.
    random_seed : int | None, optional
        Random seed for sampling weights when n_samples is specified, by default 42.

    Returns
    -------
    dict[str, list[Weights]]
        A dictionary mapping target names to lists of weight samples.
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
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Summarizes a BEAST log file by computing median, ESS, and HDI for target columns.

    Parameters
    ----------
    log_file : str | Path
        Path to the BEAST log file.
    target_columns : list[str]
        List of column names to summarize.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.
    hdi_prob : float, optional
        Probability mass for the highest density interval, by default 0.95.
    job_id : str | None, optional
        SLURM job ID for retrieving job metadata, by default None.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the summary statistics for each target column.
    """
    log = read_log_file(log_file, burn_in=burn_in)[target_columns]
    summary: dict[str, Any] = {"id": Path(log_file).stem, "n_samples": len(log)}
    for column in log.columns:
        summary[f"{column}{MEDIAN_POSTFIX}"] = log[column].median()
        summary[f"{column}{ESS_POSTFIX}"] = az.ess(np.array(log[column]))  # pyright: ignore
        lower, upper = az.hdi(np.array(log[column]), hdi_prob)  # pyright: ignore
        summary[f"{column}{LOWER_POSTFIX}"] = lower
        summary[f"{column}{UPPER_POSTFIX}"] = upper
    if job_id is not None:
        summary.update(get_job_metadata(job_id))
    return summary


def summarize_logs_dir(
    logs_dir: str | Path,
    target_columns: list[str],
    burn_in: int | float = 0.1,
    hdi_prob: float = 0.95,
    job_ids: dict[str, str] | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Summarizes all BEAST log files in a directory.

    Parameters
    ----------
    logs_dir : str | Path
        Directory containing BEAST log files.
    target_columns : list[str]
        List of column names to summarize.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.
    hdi_prob : float, optional
        Probability mass for the highest density interval, by default 0.95.
    job_ids : dict[str, str] | None, optional
        Mapping of log file IDs to SLURM job IDs for retrieving job metadata, by default None.
    n_jobs : int, optional
        Number of parallel jobs to use, by default -1 (use all available cores).

    Returns
    -------
    pd.DataFrame
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
) -> list[dict[str, list[Weights]]]:
    """
    Reads BELLA weights from all BEAST log files in a directory.

    Parameters
    ----------
    logs_dir : str | Path
        Directory containing BEAST log files.
    n_samples : int | None, optional
        Number of weight samples to return per log file, by default 100.
        If None, returns all available samples after burn-in.
    burn_in : int | float, optional
        If int, number of initial samples to discard.
        If float, fraction of samples to discard, by default 0.1.
    random_seed : int | None, optional
        Random seed for sampling weights when n_samples is specified, by default 42.
    n_jobs : int, optional
        Number of parallel jobs to use, by default -1 (use all available cores).

    Returns
    -------
    list[dict[str, list[Weights]]]
        A list of dictionaries mapping target names to lists of weight samples for each log file.
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
