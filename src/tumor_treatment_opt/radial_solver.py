from __future__ import annotations

from functools import lru_cache
from math import ceil, pi
from typing import Any

import numpy as np
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    IfPos,
    Integrate,
    LinearForm,
    Mesh,
    dx,
    exp,
    grad,
    x,
)
from ngsolve.meshes import Make1DMesh

from .config import Config
from .model import (
    BurdenTimeSeries,
    TreatmentSchedule,
    calculate_outcomes,
    gaussian_initial_conditions,
)
from .solver import (
    InitialCondition,
    FieldSnapshot,
    SimulationResult,
    _NEGATIVE_MASS_TOLERANCE,
    _time_grid,
    _copy_field
)

_RADIAL_DX = dx(bonus_intorder=2)


def build_1D_mesh(config: Config) -> Mesh:
    """Uniform radial mesh on [0, R] with elements of at most ``mesh_max_size``.

    The refinement settings of the 3D mesh are ignored: in 1D a uniformly fine
    mesh is cheap, so set ``mesh_max_size`` small for radial runs.
    """

    return _build_1D_mesh(config.domain_radius, config.mesh_max_size)


@lru_cache(maxsize=4)
def _build_1D_mesh(radius: float, max_size: float) -> Mesh:
    n_elements = ceil(radius / max_size)
    return Make1DMesh(n_elements, mapping=lambda t: radius * t)


def _ball_integrals(total: GridFunction, mesh: Mesh, config: Config) -> np.ndarray:
    """I(r_i) = integral of s^2 N(s) from 0 to every node r_i, exact for any element order."""

    element_integrals = Integrate(
        x * x * total, mesh, order=config.population_element_order + 2, element_wise=True
    )
    return np.concatenate(([0.0], np.cumsum(element_integrals.NumPy())))


def radial_von_mises(
    total: GridFunction, mesh: Mesh, r: np.ndarray, config: Config
) -> np.ndarray:
    """Closed-form von Mises stress at the nodes."""

    prefactor = (
        config.force_coupling
        * (1.0 - 2.0 * config.poisson_ratio)
        / (1.0 - config.poisson_ratio)
    )
    density = total.vec.FV().NumPy()[: mesh.nv]
    ball_integrals = _ball_integrals(total, mesh, config)
    stress = np.zeros(mesh.nv)
    # At r = 0 the formula is 0/0; its limit is zero because the ball mean tends to N(0).
    inner = r > 0.0
    ball_mean = 3.0 * ball_integrals[inner] / r[inner] ** 3
    stress[inner] = prefactor * np.abs(density[inner] - ball_mean)
    return stress


def radial_displacement(
    total: GridFunction, mesh: Mesh, r: np.ndarray, config: Config
) -> np.ndarray:
    """Radial displacement u_r at the nodes."""

    p_wave_modulus = (
        2.0 * config.shear_modulus
        * (1.0 - config.poisson_ratio)
        / (1.0 - 2.0 * config.poisson_ratio)
    )
    ball_integrals = _ball_integrals(total, mesh, config)
    displacement = np.zeros(mesh.nv)
    inner = r > 0.0
    displacement[inner] = (
        config.force_coupling
        / (p_wave_modulus * r[inner] ** 2)
        * (ball_integrals[inner] - r[inner] ** 3 / config.domain_radius**3 * ball_integrals[-1])
    )
    return displacement

def _advance_populations(
    sensitive_effective: GridFunction,
    resistant_effective: GridFunction,
    sensitive_guess: GridFunction,
    resistant_guess: GridFunction,
    diffusion_sensitive: Any,
    diffusion_resistant: Any,
    concentration: float,
    step_size: float,
    config: Config,
    nonlinear_tolerance: float,
    nonlinear_max_iterations: int,
) -> tuple[GridFunction, GridFunction]:
    """Take one backward-Euler or BDF-2 step using a fixed-point reaction solve or semi implicit method."""
    space = sensitive_effective.space
    trial, test = space.TnT()
    radial_weight = x * x

    sensitive_rhs = LinearForm(space)
    sensitive_rhs += sensitive_effective * test / step_size * radial_weight * _RADIAL_DX
    sensitive_rhs.Assemble()
    resistant_rhs = LinearForm(space)
    resistant_rhs += resistant_effective * test / step_size * radial_weight * _RADIAL_DX
    resistant_rhs.Assemble()

    semi_implicit = config.time_stepping == "semi_implicit"
    iterations = 1 if semi_implicit else nonlinear_max_iterations
    sensitive_iterate = sensitive_guess
    resistant_iterate = resistant_guess

    for _ in range(iterations):
        crowding = 1.0 - (sensitive_iterate + resistant_iterate) / config.carrying_capacity
        sensitive_rate = (
            config.growth_rate_sensitive
            * crowding
            * (1.0 - config.treatment_strength * concentration)
            - config.turnover_rate
        )
        resistant_rate = config.growth_rate_resistant * crowding - config.turnover_rate

        sensitive_form = BilinearForm(space)
        sensitive_form += (
            (1.0 / step_size - sensitive_rate) * trial * test
            + diffusion_sensitive * grad(trial) * grad(test)
        ) * radial_weight * _RADIAL_DX
        sensitive_form.Assemble()

        resistant_form = BilinearForm(space)
        resistant_form += (
            (1.0 / step_size - resistant_rate) * trial * test
            + diffusion_resistant * grad(trial) * grad(test)
        ) * radial_weight * _RADIAL_DX
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
            (sensitive_iterate.vec.FV().NumPy(), resistant_iterate.vec.FV().NumPy())
        )
        new_values = np.concatenate(
            (sensitive_new.vec.FV().NumPy(), resistant_new.vec.FV().NumPy())
        )
        relative_change = np.linalg.norm(new_values - old_values) / max(
            np.linalg.norm(new_values), 1.0e-14
        )
        sensitive_iterate = sensitive_new
        resistant_iterate = resistant_new

        if semi_implicit or relative_change <= nonlinear_tolerance:
            return sensitive_new, resistant_new

    raise RuntimeError(
        "population iteration did not converge; reduce time_step or increase "
        "nonlinear_max_iterations"
    )

def _burden(field: GridFunction, mesh: Mesh, config: Config) -> float:
    """Integral of a radial density over the full ball."""

    order = config.population_element_order + 2
    return float(4.0 * pi * Integrate(field * x * x, mesh, order=order))

def _check_nonnegative(field: GridFunction, mesh: Mesh, config: Config, name: str, time: float) -> None:
    negative_mass = _burden(IfPos(-field, -field, 0.0), mesh, config)
    total_mass = abs(_burden(field, mesh, config))
    if negative_mass > _NEGATIVE_MASS_TOLERANCE * max(total_mass, 1.0):
        raise RuntimeError(
            f"{name} density became negative at t={time:.6g}; reduce mesh_max_size "
            "relative to initial_tumor_width"
        )


def solve_radial_model(
    config: Config,
    schedule: TreatmentSchedule,
    *,
    mesh: Mesh | None = None,
    initial_condition: InitialCondition = gaussian_initial_conditions,
    save_snapshots: bool = False,
    nonlinear_tolerance: float = 1.0e-8,
    nonlinear_max_iterations: int = 25,
) -> SimulationResult:
    """Solve the spherically symmetric model on ``[0, T]`` and calculate paper outcomes.

    Stepping mirrors :func:`.solver.solve_model`, but on the radial interval
    ``[0, R]`` with every integral weighted by r^2, and with the von Mises stress
    taken from its closed form instead of an elasticity solve. Burdens include
    the factor 4 pi, so they compare directly with the 3D solver. The initial
    condition is evaluated as ``initial_condition(r, 0, 0, ...)`` and must be
    radially symmetric. Mass lumping is not supported in 1D.
    """


    schedule.validate(config)
    if config.lumping:
        raise ValueError("the radial solver does not support lumping; set lumping=False")
    if nonlinear_tolerance <= 0.0:
        raise ValueError("nonlinear_tolerance must be positive")
    if nonlinear_max_iterations < 1:
        raise ValueError("nonlinear_max_iterations must be at least one")

    mesh = build_1D_mesh(config) if mesh is None else mesh
    population_space = H1(mesh, order=config.population_element_order)
    sensitive = GridFunction(population_space, name="sensitive density")
    resistant = GridFunction(population_space, name="resistant density")
    sensitive_initial, resistant_initial = initial_condition(x, 0.0, 0.0, config, exponential=exp)

    sensitive.Interpolate(sensitive_initial)
    resistant.Interpolate(resistant_initial)

    times = _time_grid(config)
    sensitive_burden = np.empty(len(times))
    resistant_burden = np.empty(len(times))
    sensitive_burden[0] = _burden(sensitive, mesh, config)
    resistant_burden[0] = _burden(resistant, mesh, config)

    snapshots: list[FieldSnapshot] = []
    if save_snapshots:
        snapshots.append(
            FieldSnapshot(0.0, _copy_field(sensitive), _copy_field(resistant))
        )

    radii = np.array([vertex.point[0] for vertex in mesh.vertices])
    total = GridFunction(population_space)
    von_mises = GridFunction(H1(mesh, order=1), name="von Mises stress")
    stress_feedback = config.mechanics_enabled and config.stress_sensitivity != 0.0

    sensitive_previous = None
    resistant_previous = None

    for step, (time, next_time) in enumerate(zip(times[:-1], times[1:]), start=1):
        step_size = float(next_time - time)
        use_bdf2 = config.time_order == "BDF2" and step > 1
        dose_here = sum(
            dose for dose_time, dose in zip(schedule.times, schedule.increments)
            if np.isclose(time, dose_time)
        )
        if use_bdf2 and dose_here > 0.0:
            # J = dose * d(sensitive reaction)/dc; must match reaction_terms in model.py.

            s_now = sensitive.vec.FV().NumPy()
            r_now = resistant.vec.FV().NumPy()
            jump = (
                -config.growth_rate_sensitive * config.treatment_strength * dose_here
                * (1.0 - (s_now + r_now) / config.carrying_capacity) * s_now
            )
            sensitive_ghost = GridFunction(population_space)
            sensitive_ghost.vec.FV().NumPy()[:] = sensitive_previous.vec.FV().NumPy() - step_size * jump
            sensitive_previous = sensitive_ghost

        if use_bdf2:
            effective_step = 2.0 * step_size / 3.0
            sensitive_effective = GridFunction(population_space)
            sensitive_effective.vec.data = (4.0 / 3.0) * sensitive.vec - (1.0 / 3.0) * sensitive_previous.vec
            resistant_effective = GridFunction(population_space)
            resistant_effective.vec.data = (4.0 / 3.0) * resistant.vec - (1.0 / 3.0) * resistant_previous.vec
            sensitive_guess = GridFunction(population_space)
            sensitive_guess.vec.data = 2.0 * sensitive.vec - sensitive_previous.vec
            resistant_guess = GridFunction(population_space)
            resistant_guess.vec.data = 2.0 * resistant.vec - resistant_previous.vec

            concentration = float(schedule.concentration(time, config)) * np.exp(
                -config.drug_decay_rate * step_size
            )
        else:
            effective_step = step_size
            sensitive_effective, resistant_effective = sensitive, resistant
            sensitive_guess, resistant_guess = sensitive, resistant
            concentration = float(schedule.concentration(0.5 * (time + next_time), config))

        if stress_feedback:
            total.vec.data = sensitive_guess.vec + resistant_guess.vec
            von_mises.vec.FV().NumPy()[:] = radial_von_mises(total, mesh, radii, config)
            diffusion_factor = exp(-config.stress_sensitivity * von_mises)
        else:
            diffusion_factor = 1.0


        sensitive_previous, resistant_previous = sensitive, resistant
        sensitive, resistant = _advance_populations(
            sensitive_effective,
            resistant_effective,
            sensitive_guess,
            resistant_guess,
            config.diffusion_sensitive * diffusion_factor,
            config.diffusion_resistant * diffusion_factor,
            concentration,
            effective_step,
            config,
            nonlinear_tolerance,
            nonlinear_max_iterations,
        )
        
        _check_nonnegative(sensitive, mesh, config, "sensitive", next_time)
        _check_nonnegative(resistant, mesh, config, "resistant", next_time)

        sensitive_burden[step] = _burden(sensitive, mesh, config)
        resistant_burden[step] = _burden(resistant, mesh, config)
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

    if config.mechanics_enabled:
        total.vec.data = sensitive.vec + resistant.vec
        von_mises.vec.FV().NumPy()[:] = radial_von_mises(total, mesh, radii, config)
        displacement = GridFunction(von_mises.space, name="radial displacement")
        displacement.vec.FV().NumPy()[:] = radial_displacement(total, mesh, radii, config)
        von_mises_stress: GridFunction | None = von_mises
        displacement_result: GridFunction | None = displacement
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
