"""Named configurations used in the paper, validation, and tests.

``_PAPER`` holds the full reference parameter set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config

# Reference parameters for the paper results.
_PAPER: dict[str, Any] = dict(
    # Geometry and time discretization
    domain_radius=1.0,
    mesh_max_size=0.1,
    mesh_refinement_radius=0.6, 
    mesh_fine_size=0.05,
    final_time=10.0,
    time_step=0.125,
    population_element_order=1,
    displacement_element_order=1,
    lumping=True,
    time_stepping="semi_implicit",
    # Tumor growth and spreading
    growth_rate_sensitive=0.5,
    growth_rate_resistant=0.3,
    turnover_rate=0.05,
    carrying_capacity=1.0,
    diffusion_sensitive=0.01,
    diffusion_resistant=0.01,
    # Reference initial tumor
    initial_peak_occupancy=0.2,
    initial_resistant_fraction=0.05,
    initial_tumor_width=0.2,
    # Treatment schedule and response
    treatment_strength=1.0,
    drug_decay_rate=0.5,
    candidate_times=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5),
    max_concentration_increment=0.5,
    concentration_increment_budget=1.5,
    # Mechanical feedback
    mechanics_enabled=True,
    shear_modulus=1.0,
    poisson_ratio=0.3,
    force_coupling=0.1,
    stress_sensitivity=1.0,
    # Progression, remission, and objective
    progression_multiplier=2.0,
    remission_fraction=0.5,
    remission_window=0.5,
    weight_end=1.0,
    weight_average=1.0,
    weight_resistant=1.0,
    # Optimization and saved output
    optimization_seed=0,
    optimization_max_iterations=100,
    optimization_population_size=15,
    optimization_tolerance=1.0e-3,
    optimization_polish=False,
    optimization_workers=1,
    field_save_stride=1,
    output_directory=Path("outputs"),
)

_PRESETS: dict[str, dict[str, Any]] = {
    # The configuration behind the paper results.
    "paper": {},
    # Fast runs.
    "quick": dict(
        mesh_max_size=0.3,
        optimization_max_iterations=2,
        optimization_population_size=2,
    ),
    # Comparison without mechanical feedback on cell spreading.
    "no_mechanics": dict(mechanics_enabled=False),
}


def preset_names() -> tuple[str, ...]:
    return tuple(_PRESETS)


def get_config(name: str = "paper", **overrides: Any) -> Config:
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(_PRESETS)}")
    unknown = set(overrides) - set(_PAPER)
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
    return Config(**{**_PAPER, **_PRESETS[name], **overrides})
