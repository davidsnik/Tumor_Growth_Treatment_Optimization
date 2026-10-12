"""Equivalence of the radial and 3D solvers for spherically symmetric problems.

The fine radial solution is the reference; the gap to 3D should shrink as the
3D mesh is refined.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from netgen.libngpy._meshing import NgException

from tumor_treatment_opt.model import TreatmentSchedule
from tumor_treatment_opt.presets import get_config
from tumor_treatment_opt.radial_solver import solve_radial_model
from tumor_treatment_opt.solver import solve_model

CASES = ("paper", "strong_mechanics")
COMMON: dict = {"mesh_refinement_radius": 0.0}
RADIAL_MESH_SIZE = 0.00125
MESH_SIZES_3D = {
    ("paper", False): (0.1, 0.07, 0.05),
    ("paper", True): (0.2, 0.14, 0.1, 0.07, 0.05),
    ("strong_mechanics", False): (0.1, 0.07),
    ("strong_mechanics", True): (0.2, 0.14, 0.1, 0.07),
}

DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                       [1, 1, 1], [-1, 1, -1]], dtype=float)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)
OUTPUT = Path("outputs") / "validation" / "radial_3D_equivalence"


def test_schedule(config) -> TreatmentSchedule:
    increments = np.zeros(len(config.candidate_times))
    increments[[0, 4, 8]] = 0.5
    return TreatmentSchedule(config.candidate_times, tuple(increments))


def radial_profile(field, mesh, radii):
    return np.array([field(mesh(r)) for r in radii])


def _evaluate(field, mesh, point):
    try:
        return field(mesh(*point))
    except NgException:
        return np.nan


def directional_profiles(field, mesh, radii):
    """Rows: directions, columns: radii. NaN where a point lies outside the mesh."""
    return np.array([[_evaluate(field, mesh, r * d) for r in radii] for d in DIRECTIONS])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        reference_config = get_config(case, mesh_max_size=RADIAL_MESH_SIZE, lumping=False, **COMMON)
        schedule = test_schedule(reference_config)
        reference = solve_radial_model(reference_config, schedule)
        radii = np.linspace(0.0, 0.99 * reference_config.domain_radius, 100)

        profiles = {
            "radii": radii,
            "density_1D": radial_profile(reference.sensitive + reference.resistant, reference.mesh, radii),
            "stress_1D": radial_profile(reference.von_mises_stress, reference.mesh, radii),
        }
        for lumping in (False, True):
            label = "lumped" if lumping else "consistent"
            for h in MESH_SIZES_3D[(case, lumping)]:
                config = get_config(case, mesh_max_size=h, lumping=lumping, **COMMON)
                result = solve_model(config, schedule)
                gap = lambda a, b: float(np.max(np.abs(a - b) / np.abs(b)))
                rows.append({
                    "case": case,
                    "lumping": lumping,
                    "h": h,
                    "n_elements": result.mesh.ne,
                    "total_gap": gap(result.burdens.total, reference.burdens.total),
                    "resistant_gap": gap(result.burdens.resistant, reference.burdens.resistant),
                    "objective_gap": abs(result.outcomes.objective - reference.outcomes.objective)
                                     / abs(reference.outcomes.objective),
                })
                profiles[f"density_3D_{label}_h{h}"] = directional_profiles(
                    result.sensitive + result.resistant, result.mesh, radii
                )
                profiles[f"stress_3D_{label}_h{h}"] = directional_profiles(
                    result.von_mises_stress, result.mesh, radii
                )
                print(f"done: {case} {label} h={h}", flush=True)
        np.savez(OUTPUT / f"profiles_{case}.npz", **profiles)

    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
