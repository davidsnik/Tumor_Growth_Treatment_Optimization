"""Verification against an exact reference: a spatially uniform tumor.

With a uniform initial density, diffusion moves nothing, the no-flux boundary
lets nothing escape, and the tissue feels no net force, so every point follows
the same pair of ODEs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import time
from ngsolve import CoefficientFunction
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


from tumor_treatment_opt.config import Config
from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.radial_solver import build_1D_mesh, solve_radial_model

PRESET = "paper"
OVERRIDES: dict = {"lumping": False, "mesh_max_size": 0.02, "mesh_fine_size": 0.02}
CASES = {
    "paper": {}
}
SCHEMES = (
    ("backward", "semi_implicit"),
    ("backward", "picard"),
    ("BDF2", "semi_implicit"),
    ("BDF2", "picard"),
)
TIME_STEPS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
QUANTITIES = ("final_total_relative", "progression_time")
OUTPUT = Path("outputs") / "validation" / "validation_radial" / "uniform_tumor_ode.csv"


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

from scipy.optimize import brentq


def reference_quantities(config: Config, schedule: TreatmentSchedule) -> dict[str, float]:
    """N(T)/N(0) and the exact progression time from the uniform ODE."""

    evaluate = reference_solution(config, schedule)
    initial_total = sum(initial_densities(config))

    def total(t: float) -> float:
        return float(evaluate(np.array([t]))[:, 0].sum())

    threshold = config.progression_multiplier * initial_total
    grid = np.linspace(0.0, config.final_time, 4001)
    totals = np.array([total(t) for t in grid])
    above = np.flatnonzero(totals >= threshold)
    progression_time = np.inf
    if len(above):
        k = above[0]
        progression_time = brentq(lambda t: total(t) - threshold, grid[k - 1], grid[k], xtol=1.0e-14)
    return {
        "final_total_relative": total(config.final_time) / initial_total,
        "progression_time": progression_time,
    }


def simulate(case: str, time_order: str, time_stepping: str, time_step: float) -> dict:
    config = get_config(
        PRESET, time_order=time_order, time_stepping=time_stepping, time_step=time_step,
        **OVERRIDES, **CASES[case],
    )
    schedule = even_schedule(config)
    mesh = build_1D_mesh(config)
    start = time.perf_counter()
    outcomes = solve_radial_model(config, schedule, mesh=mesh, initial_condition=uniform_initial_condition).outcomes
    seconds = time.perf_counter() - start
    return {
        "case": case,
        "time_order": time_order,
        "time_stepping": time_stepping,
        "dt": time_step,
        "final_total_relative": outcomes.final_total_relative,
        "progression_time": outcomes.progression_time,
        "seconds": seconds,
    }

def main() -> None:
    base = get_config(PRESET, **OVERRIDES)
    schedule = even_schedule(base)
    reference = reference_quantities(base, schedule)

    rows = []
    for case in CASES:
        for time_order, time_stepping in SCHEMES:
            scheme_rows = [
                    simulate(case, time_order, time_stepping, time_step) for time_step in TIME_STEPS
                ]
            dt = np.array(TIME_STEPS)
            for quantity in QUANTITIES:
                values = np.array([row[quantity] for row in scheme_rows])
                errors = np.abs(values - reference[quantity])
                error_orders = np.log(errors[:-1] / errors[1:]) / np.log(dt[:-1] / dt[1:])
                for index, row in enumerate(scheme_rows):
                    row[f"{quantity}_error"] = errors[index]
                    row[f"{quantity}_error_order"] = error_orders[index - 1] if index > 0 else np.nan
            rows.extend(scheme_rows)
            print(f"done: {case} {time_order} {time_stepping}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
