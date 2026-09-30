"""Verification against an exact reference: a spatially uniform tumor.

With a uniform initial density, diffusion moves nothing, the no-flux boundary
lets nothing escape, and the tissue feels no net force, so every point follows
the same pair of ODEs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from ngsolve import CoefficientFunction
from scipy.integrate import solve_ivp

from tumor_treatment_opt.config import Config
from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.solver import solve_model

PRESET = "paper"
OVERRIDES: dict = {"mesh_max_size": 1.0}
TIME_STEPS = (0.5, 0.25, 0.125, 0.0625, 0.03125)
OUTPUT = Path("outputs") / "validation" / "uniform_tumor_ode.csv"


def initial_densities(config: Config) -> tuple[float, float]:
    total = config.carrying_capacity * config.initial_peak_occupancy
    resistant = config.initial_resistant_fraction * total
    return total - resistant, resistant


def uniform_initial_condition(x, y, z, config: Config, exponential=None):
    sensitive, resistant = initial_densities(config)
    return CoefficientFunction(sensitive), CoefficientFunction(resistant)


def even_schedule(config: Config) -> TreatmentSchedule:
    n_doses = len(config.candidate_times)
    dose = min(config.max_concentration_increment, config.concentration_increment_budget / n_doses)
    schedule = TreatmentSchedule(config.candidate_times, (dose,) * n_doses)
    peak = float(np.max(schedule.concentration(config.candidate_times, config)))
    if peak > 1.0:
        schedule = TreatmentSchedule(config.candidate_times, (0.999 * dose / peak,) * n_doses)
    return schedule


def reference_solution(config: Config, schedule: TreatmentSchedule):
    dose_times = np.asarray(schedule.times, dtype=float)
    doses = np.asarray(schedule.increments, dtype=float)
    boundaries = np.unique(np.concatenate(([0.0], dose_times, [config.final_time])))

    pieces = []
    state = np.array(initial_densities(config))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        active = dose_times <= start

        def rhs(t, y, active=active):
            sensitive, resistant = y
            u = np.sum(doses[active] * np.exp(-config.drug_decay_rate * (t - dose_times[active])))
            crowding = 1.0 - (sensitive + resistant) / config.carrying_capacity
            return [
                config.growth_rate_sensitive * crowding * (1.0 - config.treatment_strength * u)
                * sensitive - config.turnover_rate * sensitive,
                config.growth_rate_resistant * crowding * resistant - config.turnover_rate * resistant,
            ]

        piece = solve_ivp(rhs, (start, end), state, method="DOP853",
                          rtol=1.0e-12, atol=1.0e-14, dense_output=True)
        pieces.append((start, end, piece.sol))
        state = piece.y[:, -1]

    def evaluate(times: np.ndarray) -> np.ndarray:
        values = np.empty((2, len(times)))
        for index, t in enumerate(times):
            for start, end, sol in pieces:
                if start <= t <= end:
                    values[:, index] = sol(t)
                    break
        return values

    return evaluate


def main() -> None:
    base = get_config(PRESET, **OVERRIDES)
    schedule = even_schedule(base)
    reference = reference_solution(base, schedule)
    initial_total = sum(initial_densities(base))

    rows = []
    for time_step in TIME_STEPS:
        config = get_config(PRESET, time_step=time_step, **OVERRIDES)
        result = solve_model(config, schedule, initial_condition=uniform_initial_condition)
        burdens = result.burdens
        volume = burdens.total[0] / initial_total

        exact_total = reference(burdens.times).sum(axis=0)
        numerical_total = burdens.total / volume
        rows.append({
            "dt": time_step,
            "final_total_relative": result.outcomes.final_total_relative,
            "final_total_relative_exact": exact_total[-1] / initial_total,
            "final_error": abs(numerical_total[-1] - exact_total[-1]) / initial_total,
            "max_error_over_time": float(np.max(np.abs(numerical_total - exact_total))) / initial_total,
        })
        print(f"done: dt={time_step}", flush=True)

    for key in ("final_error", "max_error_over_time"):
        for index, row in enumerate(rows):
            if index >= 1:
                row[f"{key}_order"] = np.log2(rows[index - 1][key] / row[key])
            else:
                row[f"{key}_order"] = np.nan

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
