"""
Functions to simulate the synchronisation of oscillators with group interactions.
"""

import numpy as np
import xgi
from scipy.integrate import solve_ivp

from .ode_rhs import rhs_23_sym_nb

__all__ = [
    "simulate_kuramoto",
]


def simulate_kuramoto(
    H,
    omega=None,
    theta_0=None,
    t_end=100,
    dt=0.01,
    rhs=None,
    integrator="explicit_euler",
    args=(),
    seed=None,
    **options,
):
    """
    Simulate the Kuramoto model on a hypergraph.

    Parameters
    ----------
    H : xgi.Hypergraph
        Hypergraph on which to simulate coupled oscillators.
    omega : float or array-like, optional
        Natural frequencies of each node. If None (default), drawn from a
        standard normal distribution.
    theta_0 : array-like, optional
        Initial phases of each node. If None (default), drawn uniformly
        on [0, 2pi[.
    t_end : float, optional
        End time of the integration. Default is 100.
    dt : float, optional
        Time step for output. Default is 0.01.
    rhs : callable, optional
        Right-hand side of the ODE. Must return a numpy.ndarray of length N.
        Its first three arguments must be `t`, `theta`, and `omega`; extra
        arguments are passed via `args`. Default is `rhs_23_sym_nb`.
    integrator : str, optional
        Integration method. Either "explicit_euler" (default) or any method
        accepted by scipy.integrate.solve_ivp.
    args : tuple, optional
        Extra arguments passed to `rhs` after `t`, `theta`, and `omega`.
        Default is ().
    seed : int, numpy.random.Generator, or None, optional
        Seed for generating `omega` and `theta_0` when they are not provided.
        Default is None.
    **options
        Additional keyword arguments passed to scipy.integrate.solve_ivp when
        `integrator` is not "explicit_euler".

    Returns
    -------
    thetas : numpy.ndarray of shape (N, n_t)
        Phases of each node at each time step.
    times : numpy.ndarray of shape (n_t,)
        Time points corresponding to each column of `thetas`.
    """
    rng = np.random.default_rng(seed)

    H = xgi.convert_labels_to_integers(H, "label")
    N = H.num_nodes

    if omega is None:
        omega = rng.standard_normal(N)

    if theta_0 is None:
        theta_0 = rng.random(N) * 2 * np.pi

    if rhs is None:
        rhs = rhs_23_sym_nb

    times = np.arange(0, t_end + dt / 2, dt)
    n_t = len(times)

    thetas = np.zeros((N, n_t))
    thetas[:, 0] = theta_0

    if integrator == "explicit_euler":
        for it in range(1, n_t):
            thetas[:, it] = thetas[:, it - 1] + dt * rhs(
                0, thetas[:, it - 1], omega, *args
            )
    else:
        solution = solve_ivp(
            fun=rhs,
            t_span=[times[0], times[-1]],
            y0=theta_0,
            t_eval=times,
            method=integrator,
            args=(omega, *args),
            **options,
        )

        thetas = solution.y
        times = solution.t

    return thetas, times
