import numpy as np
import pandas as pd

from bella_companion.metrics import (
    CoefficientOfVariation,
    Coverage,
    MeanESSPerHour,
    NormalizedMAE,
)
from bella_companion.targets import SkylineTarget


def test_norm_mae_on_skyline_target():
    summaries = pd.DataFrame(
        {
            "targetSPi0.median": [2.0, 2.0, 0.0],
            "targetSPi1.median": [3.0, 2.0, 0.0],
        }
    )
    target = SkylineTarget(id="targetSP", skyline=np.array([2.0, 3.0]))
    metric = NormalizedMAE()
    result = metric(summaries, target)
    expected = np.array(
        [
            [0.0, 0.0],
            [0.0, 1 / 2.5],
            [2 / 2.5, 3 / 2.5],
        ]
    )
    assert np.allclose(result, expected)


def test_coverage_on_skyline_target():
    summaries = pd.DataFrame(
        {
            "targetSPi0.lower": [1.0, 2.0, 3.0, 0.5],
            "targetSPi0.upper": [3.0, 4.0, 5.0, 1.0],
            "targetSPi1.lower": [2.0, 3.0, 4.0, 3.0],
            "targetSPi1.upper": [4.0, 5.0, 6.0, 4.0],
        }
    )
    target = SkylineTarget(id="targetSP", skyline=np.array([2.5, 3.5]))
    metric = Coverage()
    result = metric(summaries, target)
    expected = np.array([0.5, 0.75])
    assert np.allclose(result, expected)


def test_coefficient_of_variation_on_skyline_target():
    summaries = pd.DataFrame(
        {
            "targetSPi0.mean": [2.0, 2.0, 2.0],
            "targetSPi0.std": [1.0, 0.5, 0.0],
            "targetSPi1.mean": [3.0, 3.0, 3.0],
            "targetSPi1.std": [1.5, 1.0, 0.5],
        }
    )
    target = SkylineTarget(id="targetSP", skyline=np.array([2.0, 3.0]))
    metric = CoefficientOfVariation()
    result = metric(summaries, target)
    expected = np.array(
        [
            [1 / 2.0, 1.5 / 3.0],
            [0.5 / 2.0, 1.0 / 3.0],
            [0.0 / 2.0, 0.5 / 3.0],
        ]
    )
    assert np.allclose(result, expected)


def test_mean_ess_per_hour_on_skyline_target():
    summaries = pd.DataFrame(
        {
            "targetSPi0.ess": [10.0, 20.0, 30.0],
            "targetSPi1.ess": [15.0, 25.0, 60.0],
            "total_hours": [1.0, 2.0, 3.0],
        }
    )
    target = SkylineTarget(id="targetSP", skyline=np.array([2.0, 3.0]))
    metric = MeanESSPerHour()
    result = metric(summaries, target)
    expected = np.array(
        [
            [10.0, 15.0],
            [10.0, 12.5],
            [10.0, 20.0],
        ]
    )
    assert np.allclose(result, expected)


def test_aggregate_coverage_single_target():
    summaries = pd.DataFrame(
        {
            "targetSPi0.lower": [1.0, 2.0, 3.0, 0.5],
            "targetSPi0.upper": [3.0, 4.0, 5.0, 1.0],
            "targetSPi1.lower": [2.0, 3.0, 4.0, 3.0],
            "targetSPi1.upper": [4.0, 5.0, 6.0, 4.0],
        }
    )
    target1 = SkylineTarget(id="targetSP", skyline=np.array([2.5, 3.5]))
    metric = Coverage()
    result = metric.aggregate(summaries, targets=[target1])
    expected = np.array(0.625)
    assert np.allclose(result, expected)


def test_aggregate_coverage_multiple_targets():
    summaries = pd.DataFrame(
        {
            "target1SPi0.lower": [1.0, 2.0, 3.0, 0.5],
            "target1SPi0.upper": [3.0, 4.0, 5.0, 1.0],
            "target1SPi1.lower": [2.0, 3.0, 4.0, 3.0],
            "target1SPi1.upper": [4.0, 5.0, 6.0, 4.0],
            "target2SPi0.lower": [1.5, 2.5, 3.0, 0.0],
            "target2SPi0.upper": [2.5, 4.0, 4.5, 0.5],
            "target2SPi1.lower": [2.5, 3.5, 4.0, 2.5],
            "target2SPi1.upper": [3.5, 5.0, 5.5, 3.5],
        }
    )
    target1 = SkylineTarget(id="target1SP", skyline=np.array([2.5, 3.5]))
    target2 = SkylineTarget(id="target2SP", skyline=np.array([3.5, 4.5]))
    metric = Coverage()
    result = metric.aggregate(summaries, targets=[target1, target2])
    expected = np.array(0.625 + 0.5) / 2
    assert np.allclose(result, expected)


def test_aggregate_per_target_norm_mae():
    summaries = pd.DataFrame(
        {
            "target1SPi0.median": [2.0, 2.0, 0.0],
            "target1SPi1.median": [3.0, 2.0, 0.0],
            "target2SPi0.median": [3.5, 2.5, 3.0],
            "target2SPi1.median": [4.5, 3.5, 4.0],
        }
    )
    target1 = SkylineTarget(id="target1SP", skyline=np.array([2.0, 3.0]))
    target2 = SkylineTarget(id="target2SP", skyline=np.array([3.5, 4.5]))
    metric = NormalizedMAE()
    result = metric.aggregate_targets(summaries, targets=[target1, target2])
    expected = np.array(
        [
            0.0,
            ((0.0 + 1.0) / 2.5 + (1.0 + 1.0) / 4.0) / 4,
            ((2.0 + 3.0) / 2.5 + (0.5 + 0.5) / 4.0) / 4,
        ]
    )
    assert np.allclose(result, expected)


def test_aggregate_per_sample_norm_mae():
    summaries = pd.DataFrame(
        {
            "target1SPi0.median": [2.0, 2.0, 0.0],
            "target1SPi1.median": [3.0, 2.0, 0.0],
            "target2SPi0.median": [3.5, 2.5, 3.0],
            "target2SPi1.median": [4.5, 3.5, 4.0],
        }
    )
    target1 = SkylineTarget(id="target1SP", skyline=np.array([2.0, 3.0]))
    target2 = SkylineTarget(id="target2SP", skyline=np.array([3.5, 4.5]))
    metric = NormalizedMAE()
    result = metric.aggregate(summaries, [target1, target2])
    expected = np.array(
        [
            0.0,
            ((0.0 + 1.0) / 2.5 + (1.0 + 1.0) / 4.0) / 4,
            ((2.0 + 3.0) / 2.5 + (0.5 + 0.5) / 4.0) / 4,
        ]
    ).mean()
    assert np.allclose(result, expected)
