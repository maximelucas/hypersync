"""
Functions to generate initial states for coupled phase oscillators.
"""

import numpy as np

__all__ = [
    "generate_q_twisted_state",
    "generate_k_clusters",
    "generate_state",
]


def generate_q_twisted_state(N, q, noise=1e-2, seed=None):
    """
    Generate a q-twisted state for N phase oscillators.

    Parameters
    ----------
    N : int
        The number of oscillators.
    q : int
        The number of twists.
    noise : float, optional
        The magnitude of gaussian noise added to the phases. Default is 1e-2.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    theta : numpy.ndarray of shape (N,)
        Array of generated phase angles representing the initial state.
    """
    rng = np.random.default_rng(seed)
    rand_phase = rng.random() * 2 * np.pi
    perturbation = noise * rng.standard_normal(N)
    theta = 2 * np.pi * q * np.arange(1, N + 1) / N + rand_phase + perturbation
    return theta


def generate_k_clusters(N, k, ps, noise=1e-2, seed=None):
    """
    Generate a k-cluster state for N phase oscillators.

    Parameters
    ----------
    N : int
        The number of oscillators.
    k : int
        The number of clusters to generate.
    ps : array-like
        The proportion of oscillators in each cluster. The number of elements
        in `ps` must be equal to `k`, and the sum of probabilities must be 1.
    noise : float, optional
        The magnitude of gaussian noise added to the phases. Default is 1e-2.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    theta : numpy.ndarray of shape (N,)
        Array of generated phase angles representing the initial state.

    Raises
    ------
    ValueError
        If `len(ps) != k` or if `ps` does not sum to 1.
    """
    rng = np.random.default_rng(seed)

    if len(ps) != k:
        raise ValueError(
            "The number of elements in ps must be equal to the number of clusters k."
        )
    if not np.isclose(sum(ps), 1):
        raise ValueError("The ps must sum to one.")

    rand_phase = rng.random() * 2 * np.pi
    perturbation = noise * rng.standard_normal(N)
    choices = rand_phase + np.linspace(0, 2 * np.pi, num=k, endpoint=False)
    theta = rng.choice(choices, size=N, p=ps) + perturbation
    return theta


def generate_state(N, kind="random", noise=1e-2, seed=None, **kwargs):
    """
    Generate initial conditions for a system of N oscillators.

    Parameters
    ----------
    N : int
        Number of oscillators in the system.
    kind : str, optional
        Kind of state to generate. Default is "random".

        * "sync": all oscillators at the same random phase,
        * "random": uniform random phases on [0, 2pi[,
        * "splay": evenly spaced on [0, 2pi[,
        * "k-cluster": random k-cluster state,
        * "q-twisted": q-twisted state.
    noise : float, optional
        Level of noise to add to the initial conditions. Default is 1e-2.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the random number generator. Default is None.
    **kwargs
        Keyword arguments passed to `generate_k_clusters` or
        `generate_q_twisted_state`.

    Returns
    -------
    theta : numpy.ndarray of shape (N,)
        Initial phases for each oscillator.

    Raises
    ------
    ValueError
        If `kind` is not one of the supported options.
    """
    rng = np.random.default_rng(seed)

    if kind == "sync":
        rand_phase = rng.random() * 2 * np.pi
        perturbation = noise * rng.standard_normal(N)
        theta = rand_phase * np.ones(N) + perturbation
    elif kind == "random":
        theta = rng.random(N) * 2 * np.pi
    elif kind == "splay":
        perturbation = noise * rng.standard_normal(N)
        theta = np.linspace(0, 2 * np.pi, num=N, endpoint=False) + perturbation
    elif kind == "k-cluster":
        theta = generate_k_clusters(N, **kwargs, noise=noise, seed=rng)
    elif kind == "q-twisted":
        theta = generate_q_twisted_state(N, **kwargs, noise=noise, seed=rng)
    else:
        raise ValueError(
            "Unknown kind. Must be one of 'sync', 'random', 'splay', 'k-cluster', 'q-twisted'."
        )

    return theta
