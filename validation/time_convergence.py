"""Self-convergence of the time-stepping under dt refinement.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tumor_treatment_opt.config import Config
from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.solver import build_spherical_mesh, solve_model

PRESET = "paper"
OVERRIDES: dict = {}
TIME_STEPS = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
QUANTITIES = ("objective", "final_total_relative")
OUTPUT = Path("outputs") / "validation" / "time_convergence.csv"   # line 21


def even_schedule(config: Config) -> TreatmentSchedule:
    """Equal doses that use the budget, scaled down if needed to keep u(t) <= 1."""

    n_doses = len(config.candidate_times)
    dose = min(config.max_concentration_increment, config.concentration_increment_budget / n_doses)
    schedule = TreatmentSchedule(config.candidate_times, (dose,) * n_doses)
    peak = float(np.max(schedule.concentration(config.candidate_times, config)))
    if peak > 1.0:
        schedule = TreatmentSchedule(config.candidate_times, (0.999 * dose / peak,) * n_doses)
    return schedule


def main() -> None:
    rows = []
    schedule = None
    for time_step in TIME_STEPS:
        config = get_config(PRESET, time_step=time_step, **OVERRIDES)
        if schedule is None:
            schedule = even_schedule(config)
        mesh = build_spherical_mesh(config)
        outcomes = solve_model(config, schedule, mesh=mesh).outcomes
        rows.append({
            "dt": time_step,
            "objective": outcomes.objective,
            "final_total_relative": outcomes.final_total_relative,
        })
        print(f"done: time_step={time_step}", flush=True)

    dt = np.array([row["dt"] for row in rows])
    for quantity in QUANTITIES:
        values = np.array([row[quantity] for row in rows])
        differences = np.abs(values[:-1] - values[1:])
        orders = np.log(differences[:-1] / differences[1:]) / np.log(dt[1:-1] / dt[2:])

        for idx in range(len(rows)):
            difference = differences[idx - 1] if idx > 0 else np.nan
            order = orders[idx - 2] if idx > 1 else np.nan
            rows[idx][f"{quantity}_difference"] = difference
            rows[idx][f"{quantity}_order"] = order

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
