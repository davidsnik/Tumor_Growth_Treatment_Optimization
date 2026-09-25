"""Treatment-schedule constraints, comparators, and optimization."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, Bounds, LinearConstraint, OptimizeResult
from functools import partial

from src.tumor_treatment_opt.config import Config
from src.tumor_treatment_opt.model import TreatmentSchedule, OutcomeMetrics
from src.tumor_treatment_opt.solver import build_spherical_mesh, solve_model


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
            return _run_cobyqa(self.config, self.starts)
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

def _default_starts(config: Config) -> list[np.ndarray]:
    """No treatment, even dosing, and front-loaded dosing, made feasible."""

    n_doses = len(config.candidate_times)
    a_max = config.max_concentration_increment
    even = np.full(n_doses, min(a_max, config.concentration_increment_budget / n_doses))
    front = np.zeros(n_doses)
    front[: (n_doses + 1) // 2] = a_max
    return [_make_feasible(x, config) for x in (np.zeros(n_doses), even, front)]

def _make_feasible(increments: np.ndarray, config: Config) -> np.ndarray:
    """Scale a schedule down just enough to satisfy the budget and u <= 1.

    Every constraint is an upper limit on a nonnegative combination of doses,
    so one common scale factor always restores feasibility. Points that are
    already feasible are returned unchanged.
    """

    x = np.clip(np.asarray(increments, dtype=float), 0.0, config.max_concentration_increment)
    scale = 1.0
    total = x.sum()
    if total > 0.0:
        scale = min(scale, (config.concentration_increment_budget - _CONSTRAINT_MARGIN) / total)
    peak = float(np.max(_concentration_matrix(config) @ x))
    if peak > 0.0:
        scale = min(scale, (1.0 - _CONSTRAINT_MARGIN) / peak)
    return x * scale


def _objective(increments: np.ndarray, config: Config) -> float:
    schedule = _make_schedule(_make_feasible(increments, config), config)

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

def _top_candidates(
    result: OptimizeResult, count: int, min_distance: float
) -> list[np.ndarray]:
    """Best distinct schedules from a DE population, best first.

    A candidate closer than ``min_distance`` to an already chosen one is
    skipped, so the starts do not all lie in the same spot.
    """

    chosen: list[np.ndarray] = []
    for index in np.argsort(result.population_energies):
        if not np.isfinite(result.population_energies[index]):
            break
        candidate = np.asarray(result.population[index], dtype=float)
        if all(np.linalg.norm(candidate - other) >= min_distance for other in chosen):
            chosen.append(candidate.copy())
        if len(chosen) == count:
            break
    return chosen


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
def _de_search(config: Config) -> OptimizeResult:
    """Run differential evolution and return SciPy's raw result.

    Besides the usual fields, the result holds the final ``population`` and
    ``population_energies`` (candidate schedules and their scores), and the
    ``history`` and ``wall_time`` added here.
    """

    parallel = config.optimization_workers != 1
    # Build the mesh before the pool starts so forked workers inherit the cache.
    build_spherical_mesh(config)
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
    result.wall_time = time.perf_counter() - start
    result.history = tuple(history)
    return result

def _to_optimization_result(
    result: OptimizeResult, config: Config, method: str
) -> OptimizationResult:
    """Simulate the best schedule of a finished search and package the run."""

    schedule = _make_schedule(_make_feasible(result.x, config), config)
    outcomes = solve_model(config, schedule, mesh=build_spherical_mesh(config)).outcomes

    return OptimizationResult(
        schedule=schedule,
        outcomes=outcomes,
        objective_value=float(result.fun),
        method=method,
        converged=bool(result.success),
        message=str(result.message),
        evaluations=int(result.nfev),
        failed_evaluations=0,
        wall_time=float(result.wall_time),
        history=tuple(result.history),
    )

def _run_de(config: Config) -> OptimizationResult:
    """Global schedule search with differential evolution."""

    return _to_optimization_result(_de_search(config), config, method="de")


def _run_cobyqa(
    config: Config,
    starts: tuple[np.ndarray, ...] | None = None,
    max_evaluations: int = 100,
) -> OptimizationResult:
    """Local schedule search with COBYQA from one or more starting schedules."""

    mesh = build_spherical_mesh(config)
    problem = build_problem(config)
    start_points = _default_starts(config) if starts is None else list(starts)
    a_max = config.max_concentration_increment

    best = None
    best_history: list[float] = []
    evaluations = 0

    start = time.perf_counter()

    for x0 in start_points:
        history: list[float] = []

        def record(intermediate_result: OptimizeResult) -> None:
            history.append(float(intermediate_result.fun))

        result = minimize(
            problem.objective,
            x0=_make_feasible(x0, config),
            method="COBYQA",
            bounds=problem.bounds,
            constraints=problem.constraints,
            callback=record,
            options={
                "maxfev": max_evaluations,
                # Step sizes relative to the dose range.
                "initial_tr_radius": 0.25 * a_max,
                "final_tr_radius": 1.0e-3 * a_max,
            },
        )
        evaluations += int(result.nfev)
        if best is None or result.fun < best.fun:
            best, best_history = result, history

    wall_time = time.perf_counter() - start

    # COBYQA may end a bit outside the constraints.
    schedule = _make_schedule(_make_feasible(best.x, config), config)
    outcomes = solve_model(config, schedule, mesh=mesh).outcomes

    return OptimizationResult(
        schedule=schedule,
        outcomes=outcomes,
        objective_value=float(best.fun),
        method="cobyqa",
        converged=bool(best.success),
        message=str(best.message),
        evaluations=evaluations,
        failed_evaluations=0,
        wall_time=wall_time,
        history=tuple(best_history),
    )


def _run_hybrid(
    config: Config,
    cheap_config: Config | None = None,
    max_evaluations: int = 100,
    n_starts: int = 3,
    min_distance: float | None = None,
) -> OptimizationResult:
    """DE on a cheap model to find promising regions, then COBYQA on the full model."""

    if cheap_config is None:
        cheap_config = (
            replace(config, mesh_max_size=2.0 * config.mesh_max_size)
            if config.lumping
            else replace(config, mechanics_enabled=False)
        )
    if cheap_config.candidate_times != config.candidate_times:
        raise ValueError("cheap_config must use the same candidate_times as config")
    if min_distance is None:
        min_distance = 0.25 * config.max_concentration_increment

    de_raw = _de_search(cheap_config)
    if de_raw.fun >= _PENALTY:
        raise RuntimeError(
            "every cheap-model evaluation failed or was invalid; choose a different cheap_config"
        )
    de_result = _to_optimization_result(de_raw, cheap_config, method="de")

    starts = _top_candidates(de_raw, count=n_starts, min_distance=min_distance)
    cobyqa_result = _run_cobyqa(config, starts=starts or None, max_evaluations=max_evaluations)

    return replace(
        cobyqa_result,
        method="hybrid",
        evaluations=de_result.evaluations + cobyqa_result.evaluations,
        wall_time=de_result.wall_time + cobyqa_result.wall_time,
        stages=(de_result, cobyqa_result),
    )

    