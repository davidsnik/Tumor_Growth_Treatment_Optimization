"""Time-scheme orders in the 3D model: backward Euler vs BDF2, both semi-implicit.

A lighter version of the radial comparison: Picard is left out (the radial study shows
it is not worth its cost) and T is shortened, since the order does not depend on T.
Two measures per run:
- self-convergence: the change in a quantity when dt is halved, and the order it implies;
- error against a Richardson limit from the two finest BDF2 runs (no extra solves).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from tumor_treatment_opt.config import Config
from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.solver import build_spherical_mesh, solve_model

PRESET = "paper"
FINAL_TIME = 5.0
DOSE = 0.0375  # per dose, as in the radial runs (paper budget 1.5 over 40 doses)
N_DOSES = int(FINAL_TIME / 0.5)
OVERRIDES: dict = {
    "final_time": FINAL_TIME,
    "candidate_times": tuple(0.5 * k for k in range(N_DOSES)),
    "concentration_increment_budget": DOSE * N_DOSES,
}
CASES = {
    "paper": {},
    "strong_mechanics": {"stress_sensitivity": 300.0, "domain_radius": 2.0},
}
SCHEMES = (
    ("backward", "semi_implicit"),
    ("BDF2", "semi_implicit"),
)
TIME_STEPS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
QUANTITIES = ("final_total_relative", "progression_time")
OUTPUT = Path("outputs") / "validation" / "validation_3D" / "time_scheme_comparison.csv"


def even_schedule(config: Config) -> TreatmentSchedule:
    """Equal doses that use the budget, scaled down if needed to keep u(t) <= 1."""

    n_doses = len(config.candidate_times)
    dose = min(config.max_concentration_increment, config.concentration_increment_budget / n_doses)
    schedule = TreatmentSchedule(config.candidate_times, (dose,) * n_doses)
    peak = float(np.max(schedule.concentration(config.candidate_times, config)))
    if peak > 1.0:
        schedule = TreatmentSchedule(config.candidate_times, (0.999 * dose / peak,) * n_doses)
    return schedule


def simulate(case: str, time_order: str, time_stepping: str, time_step: float) -> dict:
    config = get_config(
        PRESET, time_order=time_order, time_stepping=time_stepping, time_step=time_step,
        **OVERRIDES, **CASES[case],
    )
    schedule = even_schedule(config)
    mesh = build_spherical_mesh(config)
    start = time.perf_counter()
    outcomes = solve_model(config, schedule, mesh=mesh).outcomes
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
    rows = []
    for case in CASES:
        runs = {}
        for time_order, time_stepping in SCHEMES:
            runs[time_order, time_stepping] = [
                simulate(case, time_order, time_stepping, time_step) for time_step in TIME_STEPS
            ]
            print(f"done: {case} {time_order} {time_stepping}", flush=True)

        # Second-order Richardson extrapolation of the two finest BDF2 runs.
        coarse, fine = runs["BDF2", "semi_implicit"][-2:]
        limit = {q: fine[q] + (fine[q] - coarse[q]) / 3.0 for q in QUANTITIES}

        dt = np.array(TIME_STEPS)
        for scheme_rows in runs.values():
            for quantity in QUANTITIES:
                values = np.array([row[quantity] for row in scheme_rows])
                differences = np.abs(values[:-1] - values[1:])
                difference_orders = np.log(differences[:-1] / differences[1:]) / np.log(dt[1:-1] / dt[2:])
                errors = np.abs(values - limit[quantity])
                error_orders = np.log(errors[:-1] / errors[1:]) / np.log(dt[:-1] / dt[1:])
                for index, row in enumerate(scheme_rows):
                    # The difference between dt and 2*dt is stored on the dt row.
                    row[f"{quantity}_difference"] = differences[index - 1] if index > 0 else np.nan
                    row[f"{quantity}_difference_order"] = difference_orders[index - 2] if index > 1 else np.nan
                    row[f"{quantity}_limit"] = limit[quantity]
                    row[f"{quantity}_error"] = errors[index]
                    row[f"{quantity}_error_order"] = error_orders[index - 1] if index > 0 else np.nan
            rows.extend(scheme_rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
