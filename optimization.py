"""Treatment-schedule constraints, comparators, and optimization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import time
from scipy.optimize import differential_evolution, Bounds, LinearConstraint, OptimizeResult
from functools import partial

from config import Config
from model import TreatmentSchedule, OutcomeMetrics
from solver import build_spherical_mesh, solve_model


_PENALTY = 1.0e6
_PENALTY_PROGRESSION = 1.0e5
_CONSTRAINT_MARGIN = 1.0e-9

### Classes

@dataclass
class TreatmentOptimizer:
    config: Config
    method: str = "hybrid"                          # "de" | "cobyqa" | "hybrid"
    cheap_config: Config | None = None              # used by "hybrid"
    starts: tuple[np.ndarray, ...] | None = None    # used by "cobyqa"

    def __post_init__(self) -> None:
        if self.method not in ("de", "cobyqa", "hybrid"):
            raise ValueError(f"unknown optimization method: {self.method!r}")

    def solve(self) -> OptimizationResult:
        if self.method == "de":
            return _run_de(self.config)
        if self.method == "cobyqa":
            return _run_cobyqa(build_problem(self.config), self.config, self.starts)
        return _run_hybrid(self.config, self.cheap_config)

@dataclass
class ScheduleProblem:
    bounds: Bounds
    constraints: list[LinearConstraint]
    objective: Callable[[np.ndarray], float]

def _make_schedule(increments: np.ndarray, config: Config) -> TreatmentSchedule:
    return TreatmentSchedule(
        times=config.candidate_times,
        increments=tuple(float(value) for value in increments),
    )  

@dataclass(frozen=True)
class OptimizationResult:
    # The answer
    schedule: TreatmentSchedule
    outcomes: OutcomeMetrics
    objective_value: float

    # How it was obtained
    method: str
    converged: bool
    message: str

    # Cost
    evaluations: int
    failed_evaluations: int
    wall_time: float

    # Optional detail
    history: tuple[float, ...] = ()
    stages: tuple[OptimizationResult, ...] = ()


### Helpers
def _concentration_matrix(config: Config) -> np.ndarray:
    times = np.asarray(config.candidate_times, dtype=float)
    elapsed = times[:, None] - times[None, :]
    return np.tril(np.exp(-config.drug_decay_rate * np.abs(elapsed)))


def _objective(increments: np.ndarray, config: Config) -> float:
    schedule = _make_schedule(increments, config)

    try:
        schedule.validate(config)
    except ValueError:
        return _PENALTY + float(np.sum(increments))

    try:
        result = solve_model(config, schedule, mesh=build_spherical_mesh(config))
    except RuntimeError:
        return _PENALTY + float(np.sum(increments))

    outcome = result.outcomes

    if outcome.progression_prevented:
        return outcome.objective

    return outcome.objective + _PENALTY_PROGRESSION

def build_problem(config: Config) -> ScheduleProblem:
    """Collect the bounds, linear constraints, and objective for one config."""

    n_doses = len(config.candidate_times)
    bounds = Bounds(
        lb=np.zeros(n_doses),
        ub=np.full(n_doses, config.max_concentration_increment),
    )
    budget_constraint = LinearConstraint(
        np.ones((1, n_doses)),
        lb=0.0,
        ub=config.concentration_increment_budget - _CONSTRAINT_MARGIN,
    )
    concentration_constraint = LinearConstraint(
        _concentration_matrix(config), lb=0.0, ub=1.0 - _CONSTRAINT_MARGIN
    )
    return ScheduleProblem(
        bounds=bounds,
        constraints=[budget_constraint, concentration_constraint],
        objective=partial(_objective, config=config),
    )

### Optimizers
def _run_de(config: Config) -> OptimizationResult:
    """Global schedule search with differential evolution."""

    parallel = config.optimization_workers != 1
    # Build the mesh before the pool starts so forked workers inherit the cache.
    mesh = build_spherical_mesh(config)
    problem = build_problem(config)

    history: list[float] = []

    def record(intermediate_result: OptimizeResult) -> None:
        history.append(float(intermediate_result.fun))

    start = time.perf_counter()
    result = differential_evolution(
        problem.objective,
        bounds=problem.bounds,
        constraints=problem.constraints,
        seed=config.optimization_seed,
        maxiter=config.optimization_max_iterations,
        popsize=config.optimization_population_size,
        tol=config.optimization_tolerance,
        polish=config.optimization_polish,
        workers=config.optimization_workers,
        updating="deferred" if parallel else "immediate",
        callback=record,
    )
    wall_time = time.perf_counter() - start

    # simulate the winner once for its outcomes.
    schedule = _make_schedule(result.x, config)
    outcomes = solve_model(config, schedule, mesh=mesh).outcomes

    return OptimizationResult(
        schedule=schedule,
        outcomes=outcomes,
        objective_value=float(result.fun),
        method="de",
        converged=bool(result.success),
        message=str(result.message),
        evaluations=int(result.nfev),
        failed_evaluations=0,
        wall_time=wall_time,
        history=tuple(history),
    )

