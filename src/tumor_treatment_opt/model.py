"""Treatment, tumor-growth, initial-condition, and outcome definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import Config


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TreatmentSchedule:
    """Administration times and their concentration increments."""

    times: tuple[float, ...]
    increments: tuple[float, ...]

    def concentration(
        self, time: float | ArrayLike, config: Config
    ) -> float | FloatArray:
        """Calculate the normalized effective drug concentration ``u(t)``."""

        evaluation_times = np.asarray(time, dtype=float)
        administration_times = np.asarray(self.times, dtype=float)
        increments = np.asarray(self.increments, dtype=float)

        elapsed = evaluation_times[..., None] - administration_times
        active = elapsed >= 0.0
        remaining = np.exp(-config.drug_decay_rate * np.maximum(elapsed, 0.0))
        concentration = np.sum(active * increments * remaining, axis=-1)

        return float(concentration) if evaluation_times.ndim == 0 else concentration

    def validate(self, config: Config) -> None:
        """Check the treatment constraints from Equation (treatment schedule)."""

        if len(self.times) != len(self.increments):
            raise ValueError("schedule times and increments must have equal lengths")
        if tuple(self.times) != tuple(config.candidate_times):
            raise ValueError("schedule times must match the configured candidate times")
        if any(increment < 0.0 for increment in self.increments):
            raise ValueError("treatment increments cannot be negative")
        if any(
            increment > config.max_concentration_increment
            for increment in self.increments
        ):
            raise ValueError("a treatment increment exceeds a_max")
        if sum(self.increments) > config.concentration_increment_budget:
            raise ValueError("the treatment schedule exceeds budget B")

        concentrations = self.concentration(self.times, config)
        if np.max(concentrations) > 1.0:
            raise ValueError("the treatment schedule produces u(t) > 1")


def reaction_terms(
    sensitive_density: Any,
    resistant_density: Any,
    concentration: Any,
    config: Config,
) -> tuple[Any, Any]:
    """Calculate the local reaction terms for both cell populations."""

    total_density = sensitive_density + resistant_density
    crowding = 1.0 - total_density / config.carrying_capacity

    sensitive_reaction = (
        config.growth_rate_sensitive
        * crowding
        * (1.0 - config.treatment_strength * concentration)
        * sensitive_density
        - config.turnover_rate * sensitive_density
    )
    resistant_reaction = (
        config.growth_rate_resistant * crowding * resistant_density
        - config.turnover_rate * resistant_density
    )
    return sensitive_reaction, resistant_reaction


def gaussian_initial_conditions(
    x: Any,
    y: Any,
    z: Any,
    config: Config,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exponential: Callable[[Any], Any] = np.exp,
) -> tuple[Any, Any]:
    """Create a Gaussian tumor with a fixed local resistant fraction."""

    center_x, center_y, center_z = center
    distance_squared = (
        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
    )
    total_density = (
        config.carrying_capacity
        * config.initial_peak_occupancy
        * exponential(
            -distance_squared / (2.0 * config.initial_tumor_width**2)
        )
    )
    resistant_density = config.initial_resistant_fraction * total_density
    sensitive_density = total_density - resistant_density
    return sensitive_density, resistant_density


@dataclass
class BurdenTimeSeries:
    """Integrated sensitive and resistant burdens from one simulation."""

    times: FloatArray
    sensitive: FloatArray
    resistant: FloatArray

    @property
    def total(self) -> FloatArray:
        return self.sensitive + self.resistant

    @property
    def resistant_fraction(self) -> FloatArray:
        return np.divide(
            self.resistant,
            self.total,
            out=np.zeros_like(self.resistant),
            where=self.total > 0.0,
        )


@dataclass(frozen=True)
class OutcomeMetrics:
    """Progression, remission, and objective values for one simulation."""

    progression_time: float
    remission_achieved: bool
    final_total_relative: float
    average_total_relative: float
    average_resistant_relative: float
    objective: float

    @property
    def progression_prevented(self) -> bool:
        return self.progression_time == inf


def calculate_outcomes(series: BurdenTimeSeries, config: Config) -> OutcomeMetrics:
    """Calculate all treatment outcomes over the study period ``[0, T]``."""

    times = np.asarray(series.times, dtype=float)
    sensitive = np.asarray(series.sensitive, dtype=float)
    resistant = np.asarray(series.resistant, dtype=float)

    if not np.isclose(times[0], 0.0):
        raise ValueError("the simulation must start at time zero")

    end_matches = np.flatnonzero(np.isclose(times, config.final_time))
    if len(end_matches) == 0:
        raise ValueError("the time series must contain the study end time T")

    end_index = int(end_matches[0]) + 1
    times = times[:end_index]
    sensitive = sensitive[:end_index]
    resistant = resistant[:end_index]
    total = sensitive + resistant

    initial_total = float(total[0])
    initial_resistant = float(resistant[0])
    if initial_total <= 0.0 or initial_resistant <= 0.0:
        raise ValueError("the initial total and resistant burdens must be positive")

    progression_threshold = config.progression_multiplier * initial_total
    crossings = np.flatnonzero(total >= progression_threshold)
    progression_time = inf
    if len(crossings):
        index = int(crossings[0])
        progression_time = float(times[index])
        if index > 0:
            fraction = (progression_threshold - total[index - 1]) / (
                total[index] - total[index - 1]
            )
            progression_time = float(
                times[index - 1] + fraction * (times[index] - times[index - 1])
            )

    remission_start = config.final_time - config.remission_window
    remission_times = np.concatenate(
        ([remission_start], times[times > remission_start])
    )
    remission_values = np.interp(remission_times, times, total) / initial_total
    remission_achieved = bool(
        np.all(remission_values <= config.remission_fraction)
    )

    final_total_relative = float(total[-1] / initial_total)
    average_total_relative = float(
        np.trapezoid(total / initial_total, times) / config.final_time
    )
    average_resistant_relative = float(
        np.trapezoid(resistant / initial_resistant, times) / config.final_time
    )
    objective = (
        config.weight_end * final_total_relative
        + config.weight_average * average_total_relative
        + config.weight_resistant * average_resistant_relative
    )

    return OutcomeMetrics(
        progression_time=progression_time,
        remission_achieved=remission_achieved,
        final_total_relative=final_total_relative,
        average_total_relative=average_total_relative,
        average_resistant_relative=average_resistant_relative,
        objective=float(objective),
    )
