from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.radial_solver import solve_radial_model
from tumor_treatment_opt.radial_solver_jax import solve_radial_model_jax

PRESET = "paper"
OVERRIDES: dict = {
    "lumping": False,
    "mesh_max_size": 0.1,
    "mesh_fine_size": 0.1,
    "final_time": 2.0,
    "time_step": 0.125,
    "candidate_times": (0.0, 0.5, 1.0, 1.5),
    "stress_sensitivity": 30.0,
}
DOSES = (0.3, 0.1, 0.25, 0.15)
TIME_ORDERS = ("backward", "BDF2")
OUTPUT = Path("outputs") / "validation" / "validation_radial" / "ngsolve_jax_equivalence.csv"


def main() -> None:
    rows = []
    for time_order in TIME_ORDERS:
        config = get_config(PRESET, time_order=time_order, **OVERRIDES)
        schedule = TreatmentSchedule(config.candidate_times, DOSES)
        reference = solve_radial_model(config, schedule)
        result = solve_radial_model_jax(config, schedule)

        print(f"\n{time_order}: final densities at every node")
        print(f"{'r':>6} {'sensitive NGSolve':>18} {'sensitive JAX':>18} {'diff':>9}"
              f" {'resistant NGSolve':>18} {'resistant JAX':>18} {'diff':>9}")
        ngsolve_sensitive = reference.sensitive.vec.FV().NumPy()
        ngsolve_resistant = reference.resistant.vec.FV().NumPy()
        for index, r in enumerate(result.nodes):
            s_ng, s_jax = ngsolve_sensitive[index], result.sensitive[index]
            r_ng, r_jax = ngsolve_resistant[index], result.resistant[index]
            print(f"{r:6.2f} {s_ng:18.12f} {s_jax:18.12f} {abs(s_ng - s_jax):9.1e}"
                  f" {r_ng:18.12f} {r_jax:18.12f} {abs(r_ng - r_jax):9.1e}")
            rows.append({
                "time_order": time_order, "kind": "node", "position": r,
                "sensitive_ngsolve": s_ng, "sensitive_jax": s_jax,
                "resistant_ngsolve": r_ng, "resistant_jax": r_jax,
            })

        print(f"\n{time_order}: burdens at every time step")
        print(f"{'t':>6} {'N/N0 NGSolve':>16} {'N/N0 JAX':>16} {'diff':>9}")
        initial_total = reference.burdens.total[0]
        for index, t in enumerate(result.burdens.times):
            total_ng = reference.burdens.total[index] / initial_total
            total_jax = result.burdens.total[index] / initial_total
            print(f"{t:6.3f} {total_ng:16.12f} {total_jax:16.12f} {abs(total_ng - total_jax):9.1e}")
            rows.append({
                "time_order": time_order, "kind": "time", "position": t,
                "sensitive_ngsolve": reference.burdens.sensitive[index],
                "sensitive_jax": result.burdens.sensitive[index],
                "resistant_ngsolve": reference.burdens.resistant[index],
                "resistant_jax": result.burdens.resistant[index],
            })

        node_difference = max(
            np.abs(ngsolve_sensitive - result.sensitive).max(),
            np.abs(ngsolve_resistant - result.resistant).max(),
        )
        burden_difference = np.abs(reference.burdens.total - result.burdens.total).max() / initial_total
        print(f"\n{time_order}: max nodal difference {node_difference:.1e}, "
              f"max burden difference {burden_difference:.1e} (relative to N0)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
