"""Run the solver chosen by ``config.model``."""

from __future__ import annotations

from typing import Any

from .config import Config
from .model import TreatmentSchedule
from .radial_solver import build_1D_mesh, solve_radial_model
from .solver import SimulationResult, build_spherical_mesh, solve_model


SOLVERS = {
    "3D": (build_spherical_mesh, solve_model),
    "radial": (build_1D_mesh, solve_radial_model),
}


def build_mesh(config: Config):
    """Build (or fetch from the cache) the mesh for ``config.model``."""
    return SOLVERS[config.model][0](config)


def simulate(config: Config, schedule: TreatmentSchedule, **options: Any) -> SimulationResult:
    """Solve the model chosen by ``config.model``.

    ``options`` go straight to the solver (``mesh``, ``initial_condition``,
    ``save_snapshots``, ``nonlinear_tolerance``, ``nonlinear_max_iterations``).
    """
    return SOLVERS[config.model][1](config, schedule, **options)
