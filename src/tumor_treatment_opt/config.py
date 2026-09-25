"""Parameters for the tumor model, solver, and treatment optimization.

Numerical values are supplied by the notebook so that the final experiment
design remains explicit. This module only defines and validates their roles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    """Complete configuration for one simulation or optimization run."""

    # Geometry and time discretization
    domain_radius: float  # Radius R of the spherical tissue domain.
    mesh_max_size: float  # Largest tetrahedron size used by Netgen.
    final_time: float  # End time T of the simulated treatment period.
    time_step: float  # Step size used to advance the coupled model.
    population_element_order: int  # Element order for Ns and Nr.
    displacement_element_order: int  # Element order for tissue displacement.
    lumping: bool # Decides whether the solver lumps the mass matrix
    time_stepping: str  # "picard" (iterate crowding to convergence) or "semi_implicit" (one linear solve per step).

    # Tumor growth and spreading
    growth_rate_sensitive: float  # Proliferation rate r_s of sensitive cells.
    growth_rate_resistant: float  # Proliferation rate r_r of resistant cells.
    turnover_rate: float  # Shared natural cell-loss rate d_T.
    carrying_capacity: float  # Local density limit theta.
    diffusion_sensitive: float  # Baseline motility D_s,0.
    diffusion_resistant: float  # Baseline motility D_r,0.

    # Reference initial tumor
    initial_peak_occupancy: float  # Peak starting density as a fraction of theta.
    initial_resistant_fraction: float  # Resistant share of the starting tumor.
    initial_tumor_width: float  # Width of the initial central density profile.

    # Treatment schedule and response
    treatment_strength: float  # Sensitive-cell response strength eta.
    drug_decay_rate: float  # Exponential concentration-decay rate beta.
    candidate_times: tuple[float, ...]  # Fixed administration times tau_l.
    max_concentration_increment: float  # Largest allowed increment a_max.
    concentration_increment_budget: float  # Total increment budget B.

    # Mechanical feedback
    mechanics_enabled: bool  # Whether stress changes tumor-cell spreading.
    shear_modulus: float  # Tissue shear modulus G.
    poisson_ratio: float  # Tissue Poisson ratio nu.
    force_coupling: float  # Strength lambda_f of the tumor-generated force.
    stress_sensitivity: float  # Strength lambda_D of diffusion suppression.

    # Progression, remission, and objective
    progression_multiplier: float  # Progression threshold relative to N(0).
    remission_fraction: float  # Burden fraction rho_rem used for remission.
    remission_window: float  # Final interval Delta that must remain in remission.
    weight_end: float  # Weight w_end for final total burden.
    weight_average: float  # Weight w_avg for average total burden.
    weight_resistant: float  # Weight w_res for average resistant burden.

    # Optimization and saved output
    optimization_seed: int  # Random seed used by the schedule optimizer.
    optimization_max_iterations: int  # Differential-evolution generation limit.
    optimization_population_size: int  # Population multiplier used by the optimizer.
    optimization_tolerance: float  # Stopping tolerance for schedule optimization.
    optimization_polish: bool  # Whether SciPy refines the best final schedule.
    optimization_workers: int  # Number of schedule evaluations run in parallel.
    field_save_stride: int  # Number of time steps between saved 3D fields.
    output_directory: Path  # Local root for results and paper artifacts.

    def __post_init__(self) -> None:
        """Reject settings that contradict the model stated in the paper."""

        finite_values = {
            "domain_radius": self.domain_radius,
            "mesh_max_size": self.mesh_max_size,
            "final_time": self.final_time,
            "time_step": self.time_step,
            "growth_rate_sensitive": self.growth_rate_sensitive,
            "growth_rate_resistant": self.growth_rate_resistant,
            "turnover_rate": self.turnover_rate,
            "carrying_capacity": self.carrying_capacity,
            "diffusion_sensitive": self.diffusion_sensitive,
            "diffusion_resistant": self.diffusion_resistant,
            "initial_peak_occupancy": self.initial_peak_occupancy,
            "initial_resistant_fraction": self.initial_resistant_fraction,
            "initial_tumor_width": self.initial_tumor_width,
            "treatment_strength": self.treatment_strength,
            "drug_decay_rate": self.drug_decay_rate,
            "max_concentration_increment": self.max_concentration_increment,
            "concentration_increment_budget": self.concentration_increment_budget,
            "shear_modulus": self.shear_modulus,
            "poisson_ratio": self.poisson_ratio,
            "force_coupling": self.force_coupling,
            "stress_sensitivity": self.stress_sensitivity,
            "progression_multiplier": self.progression_multiplier,
            "remission_fraction": self.remission_fraction,
            "remission_window": self.remission_window,
            "weight_end": self.weight_end,
            "weight_average": self.weight_average,
            "weight_resistant": self.weight_resistant,
            "optimization_tolerance": self.optimization_tolerance,
        }
        for name, value in finite_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if any(not isfinite(time) for time in self.candidate_times):
            raise ValueError("candidate_times must be finite")

        if self.domain_radius <= 0.0:
            raise ValueError("domain_radius must be positive")
        if self.mesh_max_size <= 0.0:
            raise ValueError("mesh_max_size must be positive")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        if not 0.0 < self.time_step <= self.final_time:
            raise ValueError("time_step must lie in (0, final_time]")
        if self.time_stepping not in ("picard", "semi_implicit"):
            raise ValueError("time_stepping must be 'picard' or 'semi_implicit'")
        if self.population_element_order < 1 or self.displacement_element_order < 1:
            raise ValueError("finite-element orders must be at least one")
        if self.population_element_order != 1 and self.lumping:
            raise ValueError("lumping is only valid for population_element_order 1")

        if self.growth_rate_sensitive <= 0.0:
            raise ValueError("growth_rate_sensitive must be positive")
        if not 0.0 < self.growth_rate_resistant <= self.growth_rate_sensitive:
            raise ValueError(
                "growth_rate_resistant must lie in (0, growth_rate_sensitive]"
            )
        if self.turnover_rate < 0.0:
            raise ValueError("turnover_rate cannot be negative")
        if self.carrying_capacity <= 0.0:
            raise ValueError("carrying_capacity must be positive")
        if self.diffusion_sensitive <= 0.0 or self.diffusion_resistant <= 0.0:
            raise ValueError("baseline diffusion coefficients must be positive")

        if not 0.0 < self.initial_peak_occupancy <= 1.0:
            raise ValueError("initial_peak_occupancy must lie in (0, 1]")
        if not 0.0 < self.initial_resistant_fraction < 1.0:
            raise ValueError("initial_resistant_fraction must lie in (0, 1)")
        if self.initial_tumor_width <= 0.0:
            raise ValueError("initial_tumor_width must be positive")

        if self.treatment_strength <= 0.0:
            raise ValueError("treatment_strength must be positive")
        if self.drug_decay_rate <= 0.0:
            raise ValueError("drug_decay_rate must be positive")
        if not self.candidate_times:
            raise ValueError("candidate_times cannot be empty")
        if any(time < 0.0 or time > self.final_time for time in self.candidate_times):
            raise ValueError("candidate_times must lie in [0, final_time]")
        if any(
            later <= earlier
            for earlier, later in zip(self.candidate_times, self.candidate_times[1:])
        ):
            raise ValueError("candidate_times must be strictly increasing")
        if not 0.0 < self.max_concentration_increment <= 1.0:
            raise ValueError("max_concentration_increment must lie in (0, 1]")
        if self.concentration_increment_budget <= 0.0:
            raise ValueError("concentration_increment_budget must be positive")

        if self.shear_modulus <= 0.0:
            raise ValueError("shear_modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie between -1 and 0.5")
        if self.force_coupling <= 0.0:
            raise ValueError("force_coupling must be positive")
        if self.stress_sensitivity < 0.0:
            raise ValueError("stress_sensitivity cannot be negative")

        if self.progression_multiplier <= 1.0:
            raise ValueError("progression_multiplier must be greater than one")
        if not 0.0 < self.remission_fraction < 1.0:
            raise ValueError("remission_fraction must lie in (0, 1)")
        if not 0.0 < self.remission_window <= self.final_time:
            raise ValueError("remission_window must lie in (0, final_time]")
        weights = (self.weight_end, self.weight_average, self.weight_resistant)
        if any(weight < 0.0 for weight in weights) or not any(weights):
            raise ValueError("objective weights must be nonnegative and not all zero")

        if self.optimization_max_iterations < 1:
            raise ValueError("optimization_max_iterations must be at least one")
        if self.optimization_population_size < 1:
            raise ValueError("optimization_population_size must be at least one")
        if self.optimization_tolerance <= 0.0:
            raise ValueError("optimization_tolerance must be positive")
        if self.optimization_workers < 1:
            raise ValueError("optimization_workers must be at least one")
        if self.field_save_stride < 1:
            raise ValueError("field_save_stride must be at least one")
        output_directory = Path(self.output_directory)
        if output_directory.is_absolute() or ".." in output_directory.parts:
            raise ValueError("output_directory must stay inside the repository")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy for saved run metadata."""

        values = asdict(self)
        values["candidate_times"] = list(self.candidate_times)
        values["output_directory"] = str(self.output_directory)
        return values
