"""
Functions to simulate the synchronisation 
of oscillators with group interactions
"""

from math import sin

import numpy as np
import xgi
from scipy.integrate import solve_ivp
from numba import jit


__all__ = [
    "rhs_ring_nb",
    "simulate_kuramoto",
]

@jit(nopython=True)
def rhs_ring_nb(t, theta, omega, k1, k2, r):
    """
    RHS

    Parameters
    ----------
    sigma : float
        Triplet coupling strength
    r : int
        Pairwise and triplet nearest neighbour ranges
    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    # triadic coupling
    idx_2 = list(range(-r, 0)) + list(range(1, r + 1))
    idx_1 = range(-r, r + 1)

    for ii in range(N):
        for jj in idx_1:  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:  # triplet
            for kk in idx_2:
                if jj < kk:  # because coupling function is symmetric in j and k
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    # x2 to count triangles in both directions
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    return (k1 / r) * pairwise + k2 / (r * (2 * r - 1)) * triplets


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
        Natural frequencies of each node. If not given, a random normal distribution with mean 0 and
        standard deviation 1 is used.
    theta_0 : array-like, optional
        Initial phases of each node. If not given, a random phase is used.
    t_end : float, optional
        End time of the integration
    dt : float, optional
        Time step for the simulation.
    integrator : str, optional
        Integration method to use. Either "explicit_euler" or any method supported by
        scipy.integrate.solve_ivp.

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
        rhs = rhs_lucas

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
