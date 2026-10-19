from functools import lru_cache
from math import pi
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np

from .config import Config
from .model import (
    BurdenTimeSeries,
    OutcomeMetrics,
    TreatmentSchedule,
    calculate_outcomes,
    gaussian_initial_conditions,
)
from .solver import InitialCondition, _NEGATIVE_MASS_TOLERANCE, _time_grid


_GAUSS_POINTS = 3 


def _reference_quadrature(n_points: int = _GAUSS_POINTS) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(n_points)
    return 0.5 * (points + 1.0), 0.5 * weights


_XI, _WEIGHTS = _reference_quadrature()
_PHI = jnp.asarray(np.stack((1.0 - _XI, _XI)))
_DPHI_DXI = jnp.array([-1.0, 1.0])


class RadialMesh(NamedTuple):
    nodes: jnp.ndarray  # (n_e + 1,) node radii
    h: jnp.ndarray  # (n_e,) element lengths, the Jacobian of r = a_e + h_e * xi
    r_q: jnp.ndarray  # (n_e, n_q) physical quadrature points
    weighted_dr: jnp.ndarray  # (n_e, n_q) w_q * h_e * r_q^2, the measure r^2 dr at each point


def build_radial_mesh(config: Config) -> RadialMesh:
    n_elements = int(np.ceil(config.domain_radius / config.mesh_max_size))
    return _mesh_from_nodes(np.linspace(0.0, config.domain_radius, n_elements + 1))


def _mesh_from_nodes(nodes: np.ndarray) -> RadialMesh:
    nodes = jnp.asarray(nodes, dtype=jnp.float64)
    h = jnp.diff(nodes)
    r_q = nodes[:-1, None] + h[:, None] * jnp.asarray(_XI)[None, :]
    weighted_dr = jnp.asarray(_WEIGHTS)[None, :] * h[:, None] * r_q**2
    return RadialMesh(nodes=nodes, h=h, r_q=r_q, weighted_dr=weighted_dr)


def at_quadrature_points(nodal: jnp.ndarray) -> jnp.ndarray:
    return nodal[:-1, None] * _PHI[0] + nodal[1:, None] * _PHI[1]

def get_matrices(
    mesh: RadialMesh, mass_coefficient: jnp.ndarray, diffusion: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Element matrices, index [e, i, j] with i the test and j the trial function.
    mass = jnp.einsum("eq,iq,jq->eij", mesh.weighted_dr * mass_coefficient, _PHI, _PHI)
    stiffness = jnp.einsum(
        "eq,i,j->eij", mesh.weighted_dr * diffusion, _DPHI_DXI, _DPHI_DXI
    ) / mesh.h[:, None, None] ** 2
    element = mass + stiffness

    # Element e couples global nodes e and e + 1.
    n_nodes = mesh.nodes.shape[0]
    diagonal = jnp.zeros(n_nodes).at[:-1].add(element[:, 0, 0]).at[1:].add(element[:, 1, 1])
    lower = jnp.concatenate((jnp.zeros(1), element[:, 1, 0]))
    upper = jnp.concatenate((element[:, 0, 1], jnp.zeros(1)))
    return lower, diagonal, upper

def radial_von_mises(total: jnp.ndarray, mesh: RadialMesh, config: Config) -> jnp.ndarray:
    prefactor = config.force_coupling * (1.0 - 2.0 * config.poisson_ratio) / (1.0 - config.poisson_ratio)

    element_integrals = jnp.sum(mesh.weighted_dr * at_quadrature_points(total), axis=1)
    ball_integrals = jnp.concatenate((jnp.zeros(1), jnp.cumsum(element_integrals)))

    inner = mesh.nodes > 0.0
    safe_radius = jnp.where(inner, mesh.nodes, 1.0)
    ball_mean = 3.0 * ball_integrals / safe_radius**3
    return jnp.where(inner, prefactor * jnp.abs(total - ball_mean), 0.0)

def tridiagonal_matvec(lower, diagonal, upper, vector):
    """(A @ v) for a tridiagonal A stored as three bands (lower[0] = upper[-1] = 0)."""
    return (
        diagonal * vector
        + upper * jnp.concatenate((vector[1:], jnp.zeros(1)))
        + lower * jnp.concatenate((jnp.zeros(1), vector[:-1]))
    )


def _solve_tridiagonal(lower, diagonal, upper, rhs):
    return jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs[:, None])[:, 0]


def _advance_population(
    sensitive_effective: jnp.ndarray,
    resistant_effective: jnp.ndarray,
    sensitive_guess: jnp.ndarray,
    resistant_guess: jnp.ndarray,
    diffusion_sensitive: jnp.ndarray,
    diffusion_resistant: jnp.ndarray,
    concentration: float,
    step_size: float,
    config: Config,
    mesh: RadialMesh,
    mass_bands: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    crowding = 1.0 - at_quadrature_points(sensitive_guess + resistant_guess) / config.carrying_capacity
    sensitive_rate = (
        config.growth_rate_sensitive * crowding * (1.0 - config.treatment_strength * concentration)
        - config.turnover_rate
    )
    resistant_rate = config.growth_rate_resistant * crowding - config.turnover_rate

    def solve(effective, rate, diffusion):
        lower, diagonal, upper = get_matrices(mesh, 1.0 / step_size - rate, diffusion)
        rhs = tridiagonal_matvec(*mass_bands, effective) / step_size
        return _solve_tridiagonal(lower, diagonal, upper, rhs)

    return (
        solve(sensitive_effective, sensitive_rate, diffusion_sensitive),
        solve(resistant_effective, resistant_rate, diffusion_resistant),
    )

class RadialJaxResult(NamedTuple):
    nodes: np.ndarray
    burdens: BurdenTimeSeries
    concentration: np.ndarray
    outcomes: OutcomeMetrics
    sensitive: np.ndarray
    resistant: np.ndarray

def _burden(field: jnp.ndarray, mesh: RadialMesh) -> jnp.ndarray:
    return 4.0 * pi * jnp.sum(mesh.weighted_dr * at_quadrature_points(field))

def _negative_fraction(field: jnp.ndarray, mesh: RadialMesh) -> jnp.ndarray:
    negative = 4.0 * pi * jnp.sum(mesh.weighted_dr * jnp.maximum(-at_quadrature_points(field), 0.0))
    return negative / jnp.maximum(jnp.abs(_burden(field, mesh)), 1.0)

def _step_tables(config: Config) -> dict[str, np.ndarray]:
    times = _time_grid(config)
    starts, ends = times[:-1], times[1:]
    dose_times = np.asarray(config.candidate_times, dtype=float)
    decay = config.drug_decay_rate

    def decayed(evaluation_times):
        elapsed = evaluation_times[:, None] - dose_times[None, :]
        return np.where(elapsed >= 0.0, np.exp(-decay * np.maximum(elapsed, 0.0)), 0.0)

    step_index = np.arange(1, len(times))
    return {
        "step_size": ends - starts,
        # Backward Euler: concentration at the interval midpoint.
        "midpoint_weights": decayed(0.5 * (starts + ends)),
        # BDF2: the value just before the end of the step, c(t_n) * exp(-k dt) (doses lie on the grid).
        "end_weights": decayed(starts) * np.exp(-decay * (ends - starts))[:, None],
        # Doses given exactly at the start of each step, for the BDF2 ghost correction.
        "dose_weights": np.isclose(starts[:, None], dose_times[None, :]).astype(float),
        "use_bdf2": (config.time_order == "BDF2") & (step_index > 1),
    }

@lru_cache(maxsize=8)
def _compiled_solver(config: Config):

    if config.time_stepping != "semi_implicit":
        raise ValueError("the JAX radial solver only supports time_stepping='semi_implicit'")
    if config.lumping:
        raise ValueError("the radial solver does not support lumping; set lumping=False")

    mesh = build_radial_mesh(config)
    mass_bands = get_matrices(mesh, jnp.ones_like(mesh.r_q), jnp.zeros_like(mesh.r_q))
    tables = {key: jnp.asarray(value) for key, value in _step_tables(config).items()}
    stress_feedback = config.mechanics_enabled and config.stress_sensitivity != 0.0

    def step(carry, inputs):
        sensitive, resistant, sensitive_previous, resistant_previous = carry
        step_size, use_bdf2, concentration_midpoint, concentration_end, dose_here = inputs

        jump = (
            -config.growth_rate_sensitive * config.treatment_strength * dose_here
            * (1.0 - (sensitive + resistant) / config.carrying_capacity) * sensitive
        )
        sensitive_previous = sensitive_previous - step_size * jump

        # BDF2 as a backward-Euler step 
        # stress use the second-order prediction 2u^n - u^{n-1}.
        sensitive_effective = jnp.where(use_bdf2, (4.0 * sensitive - sensitive_previous) / 3.0, sensitive)
        resistant_effective = jnp.where(use_bdf2, (4.0 * resistant - resistant_previous) / 3.0, resistant)
        sensitive_guess = jnp.where(use_bdf2, 2.0 * sensitive - sensitive_previous, sensitive)
        resistant_guess = jnp.where(use_bdf2, 2.0 * resistant - resistant_previous, resistant)
        effective_step = jnp.where(use_bdf2, 2.0 * step_size / 3.0, step_size)
        concentration = jnp.where(use_bdf2, concentration_end, concentration_midpoint)

        if stress_feedback:
            stress = radial_von_mises(sensitive_guess + resistant_guess, mesh, config)
            diffusion_factor = jnp.exp(-config.stress_sensitivity * at_quadrature_points(stress))
        else:
            diffusion_factor = jnp.ones_like(mesh.r_q)

        sensitive_new, resistant_new = _advance_population(
            sensitive_effective, resistant_effective, sensitive_guess, resistant_guess,
            config.diffusion_sensitive * diffusion_factor, config.diffusion_resistant * diffusion_factor,
            concentration, effective_step, config, mesh, mass_bands,
        )
        outputs = (
            _burden(sensitive_new, mesh),
            _burden(resistant_new, mesh),
            jnp.maximum(_negative_fraction(sensitive_new, mesh), _negative_fraction(resistant_new, mesh)),
        )
        return (sensitive_new, resistant_new, sensitive, resistant), outputs

    @jax.jit
    def run(increments, sensitive_initial, resistant_initial):
        inputs = (
            tables["step_size"],
            tables["use_bdf2"],
            tables["midpoint_weights"] @ increments,
            tables["end_weights"] @ increments,
            tables["dose_weights"] @ increments,
        )
        carry = (sensitive_initial, resistant_initial, sensitive_initial, resistant_initial)
        (sensitive, resistant, _, _), (sensitive_burden, resistant_burden, negative) = jax.lax.scan(
            step, carry, inputs
        )
        sensitive_burden = jnp.concatenate((_burden(sensitive_initial, mesh)[None], sensitive_burden))
        resistant_burden = jnp.concatenate((_burden(resistant_initial, mesh)[None], resistant_burden))
        return sensitive_burden, resistant_burden, sensitive, resistant, negative

    return mesh, run


def solve_radial_model_jax(
    config: Config,
    schedule: TreatmentSchedule,
    *,
    mesh: RadialMesh | None = None,
    initial_condition: InitialCondition = gaussian_initial_conditions,
) -> RadialJaxResult:
    """Solve the spherically symmetric model on [0, T] with the JAX solver and calculate outcomes.
    """

    schedule.validate(config)
    radial_mesh, run = _compiled_solver(config)
    nodes = np.asarray(radial_mesh.nodes)
    sensitive_initial, resistant_initial = initial_condition(nodes, 0.0, 0.0, config, exponential=np.exp)
    sensitive_initial = np.broadcast_to(np.asarray(sensitive_initial, dtype=float), nodes.shape)
    resistant_initial = np.broadcast_to(np.asarray(resistant_initial, dtype=float), nodes.shape)

    sensitive_burden, resistant_burden, sensitive, resistant, negative = run(
        jnp.asarray(schedule.increments, dtype=float),
        jnp.asarray(sensitive_initial),
        jnp.asarray(resistant_initial),
    )
    times = _time_grid(config)
    negative = np.asarray(negative)
    if np.any(negative > _NEGATIVE_MASS_TOLERANCE):
        first = int(np.argmax(negative > _NEGATIVE_MASS_TOLERANCE))
        raise RuntimeError(
            f"a density became negative at t={times[first + 1]:.6g}; reduce mesh_max_size "
            "relative to initial_tumor_width"
        )

    burdens = BurdenTimeSeries(
        times=times,
        sensitive=np.asarray(sensitive_burden),
        resistant=np.asarray(resistant_burden),
    )
    return RadialJaxResult(
        nodes=nodes,
        burdens=burdens,
        concentration=np.asarray(schedule.concentration(times, config)),
        outcomes=calculate_outcomes(burdens, config),
        sensitive=np.asarray(sensitive),
        resistant=np.asarray(resistant),
    )