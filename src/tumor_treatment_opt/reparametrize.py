from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import numpy as np
from functools import partial

from .config import Config

_CONCENTRATION_MARGIN = 1.0e-9

@partial(jax.jit, static_argnames="config")
def doses_from_fractions(fractions: jnp.ndarray, config: Config) -> jnp.ndarray:
    times = jnp.asarray(config.candidate_times, dtype=float)
    gaps = jnp.diff(times, prepend=times[0])

    def step(concentration_after, inputs):
        fraction, gap = inputs
        concentration_before = concentration_after * jnp.exp(-config.drug_decay_rate * gap)
        headroom = jnp.minimum(
            config.max_concentration_increment, 1.0 - _CONCENTRATION_MARGIN - concentration_before
        )
        dose = fraction * headroom
        return concentration_before + dose, dose

    _, doses = jax.lax.scan(step, 0.0, (jnp.asarray(fractions, dtype=float), gaps))
    return doses

def doses_from_fractions_numpy(fractions: np.ndarray, config: Config) -> np.ndarray:
    """NumPy version of doses_from_fractions, so the NGSolve paths never run JAX."""

    times = np.asarray(config.candidate_times, dtype=float)
    gaps = np.diff(times, prepend=times[0])
    doses = np.zeros(len(times))
    concentration = 0.0
    for k, (fraction, gap) in enumerate(zip(np.asarray(fractions, dtype=float), gaps)):
        concentration *= np.exp(-config.drug_decay_rate * gap)
        headroom = min(config.max_concentration_increment, 1.0 - _CONCENTRATION_MARGIN - concentration)
        doses[k] = fraction * headroom
        concentration += doses[k]
    return doses


def fractions_from_doses(increments: np.ndarray, config: Config) -> np.ndarray:
    times = np.asarray(config.candidate_times, dtype=float)
    gaps = np.diff(times, prepend=times[0])
    fractions = np.zeros(len(times))
    concentration = 0.0
    for k, (dose, gap) in enumerate(zip(np.asarray(increments, dtype=float), gaps)):
        concentration *= np.exp(-config.drug_decay_rate * gap)
        headroom = min(config.max_concentration_increment, 1.0 - _CONCENTRATION_MARGIN - concentration)
        fractions[k] = dose / headroom if headroom > 0.0 else 0.0
        concentration += dose
    return np.clip(fractions, 0.0, 1.0)