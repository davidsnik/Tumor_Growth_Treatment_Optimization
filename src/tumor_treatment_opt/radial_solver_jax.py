from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from .config import Config


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
