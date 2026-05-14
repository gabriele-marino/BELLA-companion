from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from jaxtyping import Float
from matplotlib.axes import Axes

from bella_companion.typings import Array, ModelID, Target


class Metric(ABC):
    @abstractmethod
    def __call__(self, summaries: pd.DataFrame, target: Target) -> Array: ...
    @abstractmethod
    def aggregate(
        self, summaries: pd.DataFrame, targets: Iterable[Target]
    ) -> float: ...
    @abstractmethod
    def plot(
        self,
        ax: Axes,
        models_summaries: Mapping[ModelID, pd.DataFrame],
        targets: Iterable[Target],
    ): ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def id(self) -> str: ...
    @property
    @abstractmethod
    def lower_is_better(self) -> bool: ...


class PerStudyMetric(Metric):
    @abstractmethod
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_target_keys"]: ...  # noqa: F821

    def aggregate(self, summaries: pd.DataFrame, targets: Iterable[Target]) -> float:
        return np.mean(
            [self(summaries, target) for target in targets], dtype=np.float64
        )

    def plot(
        self,
        ax: Axes,
        models_summaries: Mapping[ModelID, pd.DataFrame],
        targets: Iterable[Target],
    ):
        ys = list(range(len(models_summaries)))
        for y, summaries in zip(ys, models_summaries.values()):
            data = self.aggregate(summaries, targets)
            ax.hlines(y=y, xmin=0, xmax=data, color="black", linewidth=1)  # pyright: ignore
            ax.plot(data, y, "o", color="black", markersize=5)  # pyright: ignore

        ax.set_yticks(ys)  # pyright: ignore
        ax.set_yticklabels(list(models_summaries))  # pyright: ignore


class PerRunMetric(Metric):
    @abstractmethod
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_runs n_target_keys"]: ...  # noqa: F722

    def aggregate_targets(
        self, summaries: pd.DataFrame, targets: Iterable[Target]
    ) -> Float[Array, "n_runs"]:  # noqa: F821
        return np.array([self(summaries, target) for target in targets]).mean(
            axis=(0, 2)
        )

    def aggregate(self, summaries: pd.DataFrame, targets: Iterable[Target]) -> float:
        return self.aggregate_targets(summaries, targets).mean()

    def plot(
        self,
        ax: Axes,
        models_summaries: Mapping[ModelID, pd.DataFrame],
        targets: Iterable[Target],
    ):
        ys = list(range(len(models_summaries)))
        for y, summaries in zip(ys, models_summaries.values()):
            values = self.aggregate_targets(summaries, targets)
            q25, q50, q75 = np.percentile(values, [25, 50, 75])  # pyright: ignore
            ax.hlines(y=y, xmin=q25, xmax=q75, color="black", linewidth=2)  # pyright: ignore
            ax.plot(q50, y, "o", color="black", markersize=4)  # pyright: ignore
        ax.set_yticks(ys)  # pyright: ignore
        ax.set_yticklabels(list(models_summaries))  # pyright: ignore


class MAPE(PerRunMetric):
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_runs n_target_keys"]:  # noqa: F722
        return (
            summaries[[f"{key}.median" for key in target.keys]]
            .sub(target.values)
            .abs()
            .div(np.mean(target.values, dtype=np.float64))
            .values
        )

    @property
    def name(self) -> str:
        return "MAPE"

    @property
    def id(self) -> str:
        return "mape"

    @property
    def lower_is_better(self) -> bool:
        return True


class Coverage(PerStudyMetric):
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_target_keys"]:  # noqa: F821
        return np.array(
            [
                (
                    (summaries[f"{key}.lower"] <= value)
                    & (summaries[f"{key}.upper"] >= value)
                ).mean()
                for key, value in target.value_map.items()
            ]
        )

    def plot(
        self,
        ax: Axes,
        models_summaries: Mapping[ModelID, pd.DataFrame],
        targets: Iterable[Target],
    ):
        super().plot(ax, models_summaries, targets)
        ax.vlines(  # pyright: ignore
            x=0.95,
            ymin=0,
            ymax=len(models_summaries) - 1,
            color="red",
            linestyle="--",
            linewidth=2,
        )

    @property
    def name(self) -> str:
        return "Coverage"

    @property
    def id(self) -> str:
        return "coverage"

    @property
    def lower_is_better(self) -> bool:
        return False


class CoefficientOfVariation(PerRunMetric):
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_runs n_target_keys"]:  # noqa: F722
        std = summaries[[f"{key}.std" for key in target.value_map]]
        mean = summaries[[f"{key}.mean" for key in target.value_map]]
        return std.div(mean.values).values

    @property
    def name(self) -> str:
        return "Coefficient of variation"

    @property
    def id(self) -> str:
        return "cv"

    @property
    def lower_is_better(self) -> bool:
        return True


class MeanESSPerHour(PerRunMetric):
    def __call__(
        self, summaries: pd.DataFrame, target: Target
    ) -> Float[Array, "n_runs n_target_keys"]:  # noqa: F722
        return (
            summaries[[f"{key}.ess" for key in target.keys]]
            .div(summaries["total_hours"], axis=0)
            .values
        )

    @property
    def name(self) -> str:
        return "Mean ESS per hour"

    @property
    def id(self) -> str:
        return "mean_ess_per_hour"

    @property
    def lower_is_better(self) -> bool:
        return False
