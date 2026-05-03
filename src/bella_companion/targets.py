from dataclasses import dataclass

from bella_companion.typings import (
    SkylineArray,
    StateMatrix,
    Target,
    TargetKey,
    TargetValue,
)


@dataclass(kw_only=True)
class SkylineTarget(Target):
    skyline: SkylineArray
    state: str | None = None

    @property
    def n_time_bins(self) -> int:
        return len(self.skyline)

    @property
    def value_map(self) -> dict[TargetKey, TargetValue]:
        return {self.get_ith_id(i): value for i, value in enumerate(self.skyline)}

    def get_ith_id(self, time_bin: int) -> TargetKey:
        id = f"{self.id}i{time_bin}"
        if self.state is not None:
            id += f"_{self.state}"
        return id


@dataclass(kw_only=True)
class MatrixTarget(Target):
    states: list[str]
    state_matrix: StateMatrix

    @property
    def value_map(self) -> dict[TargetKey, TargetValue]:
        return {
            f"{self.id}{s1}_to_{s2}": self.state_matrix[i][j]
            for i, s1 in enumerate(self.states)
            for j, s2 in enumerate([s for s in self.states if s != s1])
        }
