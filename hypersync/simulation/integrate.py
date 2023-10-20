"""
Functions to simulate the synchronisation 
of oscillators with group interactions
"""

import numpy as np
import xgi
from scipy.integrate import solve_ivp

from .ode_rhs import rhs_ring_nb

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
    args=None,
    t_eval=False,
    **options
):
    """
    Simulate the Kuramoto model on a hypergraph with links and triangles.

    Parameters
    ----------
    H : Hypergraph
        Hypergraph on which to simulate coupled oscillators
    omega : float or array-like, optional
        Natural frequencies of each node. If None (default), a random normal distribution 
        with mean 0 and standard deviation 1 is used.
    theta_0 : array-like, optional
        Initial phases of each node. If None (default), random phases are drawn uniformly
        on [0, 2pi[.
    t_end : float, optional
        End time of the integration. (Default: 100)
    dt : float, optional
        Time step for the simulation. (Default: 0.01)
    rhs : function, optional
        Function that defines the rhs of the ODE to integrate. Must output an np.ndarray
        of length H.num_nodes. Its first three arguments should be `t`, `theta`, and `omega`.
        Other arguments coming after those can be specified via `args`.(Default: rhs_ring_nb)
    integrator : str, optional
        Integration method to use. Either "explicit_euler" (default) or any method supported by
        scipy.integrate.solve_ivp.
    args : tuple
        Arguments to pass to the `rhs` function (other that `t`, `theta`, and `omega`).
        (Default: None, which raises an error).
    **options: 
        Additional keyword arguments to be passed to scipy's solve_ivp in case `integrator`
        is not "explicit_euler".

    Returns
    -------
    thetas : array-like
        Phases of each node at each time step.
    times : array-like
        Time points for each time step.
    """

    H = xgi.convert_labels_to_integers(H, "label")
    N = H.num_nodes

    if omega is None:
        omega = np.random.normal(0, 1, N)

    if theta_0 is None:
        theta_0 = np.random.random(N) * 2 * np.pi

    if rhs is None:
        rhs = rhs_ring_nb

    times = np.arange(0, t_end + dt / 2, dt)
    n_t = len(times)

    t_eval = None if not t_eval else times

    thetas = np.zeros((N, n_t))
    thetas[:, 0] = theta_0

    if integrator == "explicit_euler":
        for it in range(1, n_t):
            thetas[:, it] = thetas[:, it - 1] + dt * rhs(
                0, thetas[:, it - 1], omega, *args
            )
    else:
        print(t_eval)
        solution = solve_ivp(
            fun=rhs,
            t_span=[times[0], times[-1]],
            y0=theta_0,
            t_eval=times,
            method=integrator,
            args=(omega, *args),
            **options
        )

        thetas = solution.y
        times = solution.t

    return thetas, times
