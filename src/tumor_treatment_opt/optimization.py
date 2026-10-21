"""Treatment-schedule constraints, comparators, and optimization."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable
from functools import lru_cache

import numpy as np
import time
import os
import jax
import jax.numpy as jnp

from scipy.optimize import differential_evolution, minimize, Bounds, OptimizeResult
from functools import partial
from ngsolve import SetNumThreads

from .config import Config
from .model import TreatmentSchedule, OutcomeMetrics, gaussian_initial_conditions
from .solver import _time_grid
from .radial_solver_jax import _compiled_solver
from .simulate import build_mesh, simulate
from .reparametrize import doses_from_fractions, fractions_from_doses, doses_from_fractions_numpy


_PENALTY = 1.0
_SMOOTHING_STAGES = ((0.03, 1.0), (0.01, 0.5), (0.003, 0.25))

### Classes

@dataclass
class TreatmentOptimizer:
    config: Config
    method: str = "hybrid"                          # "de" | "cobyqa" | "hybrid" | "lbfgs"
    cheap_config: Config | None = None              # used by "hybrid"
    starts: tuple[np.ndarray, ...] | None = None    # used by "cobyqa"

    def __post_init__(self) -> None:
        if self.method not in ("de", "cobyqa", "hybrid", "lbfgs"):
            raise ValueError(f"unknown optimization method: {self.method!r}")


    def solve(self) -> OptimizationResult:
        if self.method == "de":
            return _run_de(self.config)
        if self.method == "cobyqa":
            return _run_cobyqa(self.config, self.starts)
        if self.method == "lbfgs":
            return _run_lbfgs(self.config, self.starts)
        return _run_hybrid(self.config, self.cheap_config)

@dataclass
class ScheduleProblem:
    bounds: Bounds
    objective: Callable[[np.ndarray], float]

def _make_schedule(fractions: np.ndarray, config: Config) -> TreatmentSchedule:
    increments = doses_from_fractions_numpy(np.clip(fractions, 0.0, 1.0), config)
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
def _default_starts(config: Config) -> list[np.ndarray]:
    """No treatment, half of the allowed dose throughout, and full dosing in the first half."""

    n_doses = len(config.candidate_times)
    front = np.zeros(n_doses)
    front[: (n_doses + 1) // 2] = 1.0
    return [np.zeros(n_doses), np.full(n_doses, 0.5), front]


def _objective(fractions: np.ndarray, config: Config) -> float:
    schedule = _make_schedule(fractions, config)
    try:
        result = simulate(config, schedule)
    except RuntimeError:
        return _PENALTY
    return -result.outcomes.objective



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
        ub=np.ones(n_doses),
    )
    return ScheduleProblem(
        bounds=bounds,
        objective=partial(_objective, config=config),
    )

def _progression_time_jax(times, total, threshold):
    """First time N(t) reaches the threshold, linearly interpolated; T if it never does.
    """

    above = total >= threshold
    progressed = jnp.any(above)
    index = jnp.maximum(jnp.argmax(above), 1)  # crossing step; argmax is 0 if none or at t=0
    before, after = total[index - 1], total[index]
    interior = progressed & (total[0] < threshold)
    rise = jnp.where(interior, after - before, 1.0)
    crossing = times[index - 1] + (threshold - before) / rise * (times[index] - times[index - 1])
    return jnp.where(interior, crossing, jnp.where(progressed, times[0], times[-1]))

def _survival_time(times, total, threshold, width, decay_time):
    """Smooth stand-in for min(t_p, T): the integral of a survival probability S(t).
    """

    steps = jnp.diff(times)
    above = jax.nn.sigmoid((total / threshold - 1.0) / width)
    exposure = jnp.concatenate((jnp.zeros(1), jnp.cumsum(0.5 * (above[1:] + above[:-1]) * steps)))
    survival = jnp.exp(-exposure / decay_time)
    return jnp.sum(0.5 * (survival[1:] + survival[:-1]) * steps)

@lru_cache(maxsize=8)
def _jax_objectives(config: Config):
    """Objectives of the dose fractions for model='radial_jax'.

    Returns (exact, smooth): ``exact(thetas)`` is -min(t_p, T)/T for a batch of schedules,
    the same as ``_objective``; ``smooth(theta, eps, tau)`` is (-t_soft/T, gradient).
    """

    if config.model != "radial_jax":
        raise ValueError("gradients need model='radial_jax'")
    mesh, run = _compiled_solver(config)
    times = jnp.asarray(_time_grid(config))
    nodes = np.asarray(mesh.nodes)
    sensitive_initial, resistant_initial = gaussian_initial_conditions(
        nodes, 0.0, 0.0, config, exponential=np.exp
    )
    sensitive_initial = jnp.asarray(sensitive_initial, dtype=float)
    resistant_initial = jnp.asarray(resistant_initial, dtype=float)

    def total_burden(fractions):
        doses = doses_from_fractions(fractions, config)
        sensitive, resistant, *_ = run(doses, sensitive_initial, resistant_initial)
        return sensitive + resistant

    def exact(fractions):
        total = total_burden(fractions)
        threshold = config.progression_multiplier * total[0]
        return -_progression_time_jax(times, total, threshold) / config.final_time

    def smooth(fractions, width, decay_time):
        total = total_burden(fractions)
        threshold = config.progression_multiplier * total[0]
        return -_survival_time(times, total, threshold, width, decay_time) / config.final_time

    return jax.jit(jax.vmap(exact)), jax.jit(jax.value_and_grad(smooth))

### Optimizers
def _de_search(config: Config) -> OptimizeResult:
    """Run differential evolution and return SciPy's raw result.

    Besides the usual fields, the result holds the final ``population`` and
    ``population_energies`` (candidate schedules and their scores), and the
    ``history`` and ``wall_time`` added here.
    """

    parallel = config.optimization_workers != 1
    # Build the mesh before the pool starts so forked workers inherit the cache.
    build_mesh(config)
    problem = build_problem(config)

    cores = os.cpu_count() or 1
    if parallel:
        workers = cores if config.optimization_workers == -1 else config.optimization_workers
        SetNumThreads(max(1, cores // workers))

    history: list[float] = []

    def record(intermediate_result: OptimizeResult) -> None:
        history.append(float(intermediate_result.fun))

    rng = np.random.default_rng(config.optimization_seed)
    n_doses = len(config.candidate_times)
    init = rng.uniform(size=(config.optimization_population_size * n_doses, n_doses))
    levels = np.linspace(0.1, 1.0, min(10, len(init)))
    init[: len(levels)] = levels[:, None]

    start = time.perf_counter()
    try:
        result = differential_evolution(
            problem.objective,
            bounds=problem.bounds,
            init=init,
            seed=config.optimization_seed,
            maxiter=config.optimization_max_iterations,
            popsize=config.optimization_population_size,
            tol=config.optimization_tolerance,
            polish=config.optimization_polish,
            workers=config.optimization_workers,
            updating="deferred" if parallel else "immediate",
            callback=record,
        )
    finally:
        if parallel:
            SetNumThreads(cores) 
    result.wall_time = time.perf_counter() - start
    result.history = tuple(history)
    return result

def _to_optimization_result(
    result: OptimizeResult, config: Config, method: str
) -> OptimizationResult:
    """Simulate the best schedule of a finished search and package the run."""

    schedule = _make_schedule(result.x, config)
    outcomes = simulate(config, schedule).outcomes

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

    build_mesh(config)

    problem = build_problem(config)
    start_points = _default_starts(config) if starts is None else list(starts)

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
            x0=np.clip(x0, 0.0, 1.0),
            method="COBYQA",
            bounds=problem.bounds,
            callback=record,
            options={
                "maxfev": max_evaluations,
                "initial_tr_radius": 0.25,
                "final_tr_radius": 1.0e-3,
            },
        )
        evaluations += int(result.nfev)
        if best is None or result.fun < best.fun:
            best, best_history = result, history

    wall_time = time.perf_counter() - start

    # COBYQA may end a bit outside the constraints.
    schedule = _make_schedule(best.x, config)
    outcomes = simulate(config, schedule).outcomes

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
        if config.model == "3D":
            cheap_config = (
                replace(config, mesh_max_size=2.0 * config.mesh_max_size,
                        mesh_fine_size=2.0 * config.mesh_fine_size)
                if config.lumping
                else replace(config, mechanics_enabled=False)
            )
        else:
            cheap_config = (
                replace(config, mesh_max_size=2.0 * config.mesh_max_size)
            )
    if cheap_config.candidate_times != config.candidate_times:
        raise ValueError("cheap_config must use the same candidate_times as config")
    if min_distance is None:
        min_distance = 0.1 * np.sqrt(len(config.candidate_times))

    de_raw = _de_search(cheap_config)
    if de_raw.fun >= _PENALTY:
        raise RuntimeError(
            "every cheap-model evaluation failed or was invalid; choose a different cheap_config"
        )
    de_result = _to_optimization_result(de_raw, cheap_config, method="de")

    print("DE run finished")

    starts = _top_candidates(de_raw, count=n_starts, min_distance=min_distance)
    cobyqa_result = _run_cobyqa(config, starts=starts or None, max_evaluations=max_evaluations)

    return replace(
        cobyqa_result,
        method="hybrid",
        evaluations=de_result.evaluations + cobyqa_result.evaluations,
        wall_time=de_result.wall_time + cobyqa_result.wall_time,
        stages=(de_result, cobyqa_result),
    )

def _lbfgs_starts(config: Config, n_random: int = 1) -> list[np.ndarray]:
    n_doses = len(config.candidate_times)
    rng = np.random.default_rng(config.optimization_seed)
    constant = [np.full(n_doses, level) for level in (0.25, 0.5, 1.0)]
    return constant + [rng.uniform(size=n_doses) for _ in range(n_random)]

def _run_lbfgs(
    config: Config,
    starts: tuple[np.ndarray, ...] | None = None,
    max_iterations: int = 500,
) -> OptimizationResult:
    """Local schedule search with L-BFGS-B and JAX gradients on a smoothed objective.

    From every start, L-BFGS-B runs once per smoothing stage, each stage starting where the
    previous one ended. Every final schedule is scored with the exact t_p; the best one is
    returned, and the per-start results are kept in ``stages``.
    """

    exact, smooth = _jax_objectives(config)
    problem = build_problem(config)
    start_points = _lbfgs_starts(config) if starts is None else list(starts)

    runs: list[OptimizationResult] = []
    start = time.perf_counter()

    for x0 in start_points:
        run_start = time.perf_counter()
        fractions = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
        history: list[float] = []
        evaluations = 0

        def record(intermediate_result: OptimizeResult) -> None:
            history.append(float(intermediate_result.fun))

        for width, decay_time in _SMOOTHING_STAGES:

            def fun(x, width=width, decay_time=decay_time):
                value, gradient = smooth(jnp.asarray(x), width, decay_time)
                return float(value), np.asarray(gradient, dtype=float)

            result = minimize(
                fun,
                x0=fractions,
                jac=True,
                method="L-BFGS-B",
                bounds=problem.bounds,
                callback=record,
                options={"maxiter": max_iterations},
            )
            fractions = np.clip(result.x, 0.0, 1.0)
            evaluations += int(result.nfev)

        value = float(exact(jnp.asarray(fractions)[None])[0])
        schedule = _make_schedule(fractions, config)
        runs.append(
            OptimizationResult(
                schedule=schedule,
                outcomes=simulate(config, schedule).outcomes,
                objective_value=value,
                method="lbfgs",
                converged=bool(result.success),
                message=str(result.message),
                evaluations=evaluations,
                failed_evaluations=0,
                wall_time=time.perf_counter() - run_start,
                history=tuple(history),
            )
        )

    best = min(runs, key=lambda run: run.objective_value)
    return OptimizationResult(
        schedule=best.schedule,
        outcomes=best.outcomes,
        objective_value=best.objective_value,
        method="lbfgs",
        converged=best.converged,
        message=best.message,
        evaluations=sum(run.evaluations for run in runs),
        failed_evaluations=0,
        wall_time=time.perf_counter() - start,
        history=best.history,
        stages=tuple(runs),
    )