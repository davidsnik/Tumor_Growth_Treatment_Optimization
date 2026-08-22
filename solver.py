"""Finite-element solver for the coupled tumor and tissue model.

The public entry point is :func:`solve_model`. A minimal run, once ``config``
has been created, is::

    schedule = TreatmentSchedule(config.candidate_times, (0.0,) * len(config.candidate_times))
    result = solve_model(config, schedule)
    print(result.outcomes.objective)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from netgen.csg import CSGeometry, Pnt, Sphere
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    Id,
    IfPos,
    InnerProduct,
    Integrate,
    L2,
    LinearForm,
    Mesh,
    Sym,
    Trace,
    VectorH1,
    dx,
    exp,
    grad,
    sqrt,
    x,
    y,
    z,
)

from config import Config
from model import (
    BurdenTimeSeries,
    OutcomeMetrics,
    TreatmentSchedule,
    calculate_outcomes,
    gaussian_initial_conditions,
)


InitialCondition = Callable[..., tuple[Any, Any]]


@dataclass
class FieldSnapshot:
    """Population fields saved at one time point."""

    time: float
    sensitive: GridFunction
    resistant: GridFunction


@dataclass
class SimulationResult:
    """Numerical fields and integrated outcomes from one simulation."""

    mesh: Mesh
    burdens: BurdenTimeSeries
    concentration: np.ndarray
    outcomes: OutcomeMetrics
    sensitive: GridFunction
    resistant: GridFunction
    displacement: GridFunction | None
    von_mises_stress: GridFunction | None
    snapshots: list[FieldSnapshot]


@dataclass
class _MechanicalProblem:
    space: Any
    displacement: GridFunction
    inverse: Any
    stress_space: Any


def build_spherical_mesh(config: Config) -> Mesh:
    """Generate the tetrahedral spherical domain used by both model parts."""

    geometry = CSGeometry()
    geometry.Add(
        Sphere(Pnt(0.0, 0.0, 0.0), config.domain_radius).bc("outer")
    )
    return Mesh(geometry.GenerateMesh(maxh=config.mesh_max_size))


def _copy_field(field: GridFunction) -> GridFunction:
    copied = GridFunction(field.space)
    copied.vec.data = field.vec
    return copied


def _time_grid(config: Config) -> np.ndarray:
    """Use steps no larger than dt and place every administration on the grid."""

    base_times = np.arange(0.0, config.final_time, config.time_step)
    proposed = np.sort(
        np.concatenate(
            (
                base_times,
                np.asarray(config.candidate_times, dtype=float),
                np.array([config.final_time]),
            )
        )
    )

    tolerance = 1.0e-12 * max(1.0, config.final_time)
    times = [float(proposed[0])]
    for time in proposed[1:]:
        if time - times[-1] > tolerance:
            times.append(float(time))
        else:
            times[-1] = float(time)
    return np.asarray(times)


def _prepare_mechanics(mesh: Mesh, config: Config) -> _MechanicalProblem:
    space = VectorH1(
        mesh,
        order=config.displacement_element_order,
        dirichlet="outer",
    )
    trial, test = space.TnT()

    def strain(field: Any) -> Any:
        return Sym(grad(field))

    lame_lambda = (
        2.0
        * config.shear_modulus
        * config.poisson_ratio
        / (1.0 - 2.0 * config.poisson_ratio)
    )

    def stress(field: Any) -> Any:
        epsilon = strain(field)
        return (
            2.0 * config.shear_modulus * epsilon
            + lame_lambda * Trace(epsilon) * Id(3)
        )

    stiffness = BilinearForm(space)
    stiffness += InnerProduct(stress(trial), strain(test)) * dx
    stiffness.Assemble()

    displacement = GridFunction(space)
    inverse = stiffness.mat.Inverse(space.FreeDofs(), inverse="sparsecholesky")
    stress_order = max(0, config.displacement_element_order - 1)
    return _MechanicalProblem(
        space=space,
        displacement=displacement,
        inverse=inverse,
        stress_space=L2(mesh, order=stress_order),
    )


def _solve_mechanics(
    problem: _MechanicalProblem,
    total_density: Any,
    config: Config,
) -> tuple[GridFunction, Any]:
    """Solve mechanical equilibrium and return von Mises stress."""

    test = problem.space.TestFunction()
    load = LinearForm(problem.space)
    load += config.force_coupling * total_density * Trace(grad(test)) * dx
    load.Assemble()
    problem.displacement.vec.data = problem.inverse * load.vec

    epsilon = Sym(grad(problem.displacement))
    lame_lambda = (
        2.0
        * config.shear_modulus
        * config.poisson_ratio
        / (1.0 - 2.0 * config.poisson_ratio)
    )
    stress = (
        2.0 * config.shear_modulus * epsilon
        + lame_lambda * Trace(epsilon) * Id(3)
    )
    deviatoric_stress = stress - Trace(stress) * Id(3) / 3.0
    von_mises = sqrt(1.5 * InnerProduct(deviatoric_stress, deviatoric_stress))
    return problem.displacement, von_mises


def _advance_populations(
    sensitive_old: GridFunction,
    resistant_old: GridFunction,
    diffusion_sensitive: Any,
    diffusion_resistant: Any,
    concentration: float,
    step_size: float,
    config: Config,
    nonlinear_tolerance: float,
    nonlinear_max_iterations: int,
) -> tuple[GridFunction, GridFunction]:
    """Take one backward-Euler step using a fixed-point reaction solve."""

    space = sensitive_old.space
    trial, test = space.TnT()
    sensitive_iterate = _copy_field(sensitive_old)
    resistant_iterate = _copy_field(resistant_old)

    sensitive_rhs = LinearForm(space)
    sensitive_rhs += sensitive_old * test / step_size * dx
    sensitive_rhs.Assemble()
    resistant_rhs = LinearForm(space)
    resistant_rhs += resistant_old * test / step_size * dx
    resistant_rhs.Assemble()

    for _ in range(nonlinear_max_iterations):
        crowding = 1.0 - (
            sensitive_iterate + resistant_iterate
        ) / config.carrying_capacity
        sensitive_rate = (
            config.growth_rate_sensitive
            * crowding
            * (1.0 - config.treatment_strength * concentration)
            - config.turnover_rate
        )
        resistant_rate = (
            config.growth_rate_resistant * crowding - config.turnover_rate
        )

        sensitive_form = BilinearForm(space)
        sensitive_form += (
            (1.0 / step_size - sensitive_rate) * trial * test
            + diffusion_sensitive * InnerProduct(grad(trial), grad(test))
        ) * dx
        sensitive_form.Assemble()

        resistant_form = BilinearForm(space)
        resistant_form += (
            (1.0 / step_size - resistant_rate) * trial * test
            + diffusion_resistant * InnerProduct(grad(trial), grad(test))
        ) * dx
        resistant_form.Assemble()

        sensitive_new = GridFunction(space)
        sensitive_new.vec.data = sensitive_form.mat.Inverse(
            space.FreeDofs(), inverse="sparsecholesky"
        ) * sensitive_rhs.vec
        resistant_new = GridFunction(space)
        resistant_new.vec.data = resistant_form.mat.Inverse(
            space.FreeDofs(), inverse="sparsecholesky"
        ) * resistant_rhs.vec

        old_values = np.concatenate(
            (
                sensitive_iterate.vec.FV().NumPy().copy(),
                resistant_iterate.vec.FV().NumPy().copy(),
            )
        )
        new_values = np.concatenate(
            (
                sensitive_new.vec.FV().NumPy(),
                resistant_new.vec.FV().NumPy(),
            )
        )
        relative_change = np.linalg.norm(new_values - old_values) / max(
            np.linalg.norm(new_values), 1.0e-14
        )
        sensitive_iterate = sensitive_new
        resistant_iterate = resistant_new

        if relative_change <= nonlinear_tolerance:
            return sensitive_new, resistant_new

    raise RuntimeError(
        "population iteration did not converge; reduce time_step or increase "
        "nonlinear_max_iterations"
    )


def _check_nonnegative(field: GridFunction, mesh: Mesh, name: str, time: float) -> None:
    negative_mass = float(Integrate(IfPos(-field, -field, 0.0), mesh))
    total_mass = abs(float(Integrate(field, mesh)))
    if negative_mass > 1.0e-10 * max(total_mass, 1.0):
        raise RuntimeError(
            f"{name} density became negative at t={time:.6g}; reduce time_step"
        )


def solve_model(
    config: Config,
    schedule: TreatmentSchedule,
    *,
    mesh: Mesh | None = None,
    initial_condition: InitialCondition = gaussian_initial_conditions,
    save_snapshots: bool = False,
    nonlinear_tolerance: float = 1.0e-8,
    nonlinear_max_iterations: int = 25,
) -> SimulationResult:
    """Solve the full model on ``[0, T]`` and calculate paper outcomes.

    On each interval, mechanics is solved from the current tumor, treatment is
    evaluated at the interval midpoint, and the two population equations are
    advanced by backward Euler. Fixed-point iteration handles their shared
    crowding term. The H1 formulation imposes the no-flux conditions naturally.

    A mesh may be supplied to reuse it across schedule evaluations. Population
    fields are always returned at the final time; intermediate fields are kept
    only when ``save_snapshots`` is true.
    """

    schedule.validate(config)
    if nonlinear_tolerance <= 0.0:
        raise ValueError("nonlinear_tolerance must be positive")
    if nonlinear_max_iterations < 1:
        raise ValueError("nonlinear_max_iterations must be at least one")

    mesh = build_spherical_mesh(config) if mesh is None else mesh
    population_space = H1(mesh, order=config.population_element_order)
    sensitive = GridFunction(population_space, name="sensitive density")
    resistant = GridFunction(population_space, name="resistant density")
    sensitive_initial, resistant_initial = initial_condition(
        x, y, z, config, exponential=exp
    )
    sensitive.Interpolate(sensitive_initial)
    resistant.Interpolate(resistant_initial)

    times = _time_grid(config)
    sensitive_burden = np.empty(len(times))
    resistant_burden = np.empty(len(times))
    sensitive_burden[0] = Integrate(sensitive, mesh)
    resistant_burden[0] = Integrate(resistant, mesh)

    snapshots: list[FieldSnapshot] = []
    if save_snapshots:
        snapshots.append(
            FieldSnapshot(0.0, _copy_field(sensitive), _copy_field(resistant))
        )

    mechanics = _prepare_mechanics(mesh, config) if config.mechanics_enabled else None
    displacement = None
    von_mises = None

    for step, (time, next_time) in enumerate(zip(times[:-1], times[1:]), start=1):
        step_size = float(next_time - time)
        if mechanics is None:
            diffusion_factor = 1.0
        else:
            displacement, von_mises = _solve_mechanics(
                mechanics, sensitive + resistant, config
            )
            diffusion_factor = (
                1.0
                if config.stress_sensitivity == 0.0
                else exp(-config.stress_sensitivity * von_mises)
            )

        # Midpoint evaluation applies a dose to intervals after, not before, it occurs.
        concentration = float(schedule.concentration(0.5 * (time + next_time), config))
        sensitive, resistant = _advance_populations(
            sensitive,
            resistant,
            config.diffusion_sensitive * diffusion_factor,
            config.diffusion_resistant * diffusion_factor,
            concentration,
            step_size,
            config,
            nonlinear_tolerance,
            nonlinear_max_iterations,
        )
        _check_nonnegative(sensitive, mesh, "sensitive", next_time)
        _check_nonnegative(resistant, mesh, "resistant", next_time)

        sensitive_burden[step] = Integrate(sensitive, mesh)
        resistant_burden[step] = Integrate(resistant, mesh)
        if save_snapshots and (
            step % config.field_save_stride == 0 or step == len(times) - 1
        ):
            snapshots.append(
                FieldSnapshot(
                    float(next_time),
                    _copy_field(sensitive),
                    _copy_field(resistant),
                )
            )

    if mechanics is not None:
        displacement, von_mises = _solve_mechanics(
            mechanics, sensitive + resistant, config
        )
        stress_field = GridFunction(mechanics.stress_space, name="von Mises stress")
        stress_field.Set(von_mises)
        von_mises_stress: GridFunction | None = stress_field
        displacement_result: GridFunction | None = _copy_field(displacement)
    else:
        von_mises_stress = None
        displacement_result = None

    burdens = BurdenTimeSeries(
        times=times,
        sensitive=sensitive_burden,
        resistant=resistant_burden,
    )
    return SimulationResult(
        mesh=mesh,
        burdens=burdens,
        concentration=np.asarray(schedule.concentration(times, config)),
        outcomes=calculate_outcomes(burdens, config),
        sensitive=sensitive,
        resistant=resistant,
        displacement=displacement_result,
        von_mises_stress=von_mises_stress,
        snapshots=snapshots,
    )
