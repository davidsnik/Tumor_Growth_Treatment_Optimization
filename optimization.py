"""Treatment-schedule constraints, comparators, and optimization."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, Bounds, LinearConstraint

from config import Config
from model import TreatmentSchedule
from solver import build_spherical_mesh, solve_model

_PENALTY = 1.0e6
_PENALTY_PROGRESSION = 1.0e5

def _make_schedule(increments: np.ndarray, config: Config) -> TreatmentSchedule:
    return TreatmentSchedule(
        times=config.candidate_times,
        increments=tuple(float(value) for value in increments),
    )

def _objective(increments: np.ndarray, config: Config, shared_mesh) -> float:
    schedule = _make_schedule(increments, config)
 
    try:
        schedule.validate(config)
    except ValueError:
        return _PENALTY + float(np.sum(increments))
 
    mesh = shared_mesh if shared_mesh is not None else build_spherical_mesh(config)
 
    try:
        result = solve_model(config, schedule, mesh=mesh)
    except RuntimeError:
        return _PENALTY + float(np.sum(increments))
 
    outcome = result.outcomes
    
    if outcome.progression_prevented:
        return outcome.objective
    
    return outcome.objective + _PENALTY_PROGRESSION


def optimize_treatment(config: Config) -> np.ndarray:
    """Gradient-free method to find the optimal doses at the increments"""
 
    n_doses = len(config.candidate_times)
    bounds = Bounds(
        lb=np.zeros(n_doses),
        ub=np.full(n_doses, config.max_concentration_increment),
    )

    budget_constraint = LinearConstraint(
        np.ones(n_doses), lb=0.0, ub=config.concentration_increment_budget
    )
 
    parallel = config.optimization_workers != 1
    shared_mesh = None if parallel else build_spherical_mesh(config)
 
    result = differential_evolution(
        _objective,
        bounds=bounds,
        args=(config, shared_mesh),
        constraints=(budget_constraint,),
        seed=config.optimization_seed,
        maxiter=config.optimization_max_iterations,
        popsize=config.optimization_population_size,
        tol=config.optimization_tolerance,
        polish=config.optimization_polish,
        workers=config.optimization_workers,
        updating="deferred" if parallel else "immediate",
    )
 
    if not result.success:
        raise RuntimeError(f"schedule optimization failed: {result.message}")
 
    return result.x
 
 
def best_schedule(config: Config) -> TreatmentSchedule:
    """Returns the final treatment schedule"""
 
    increments = optimize_treatment(config)
    schedule = _make_schedule(increments, config)
    schedule.validate(config)
    return schedule