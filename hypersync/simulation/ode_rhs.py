"""
Righ-hand-side functions of ODEs for simulating synchronisation 
of oscillators with group interactions
"""

from math import sin

import numpy as np
import xgi
from numba import jit

__all__ = [
    "rhs_ring_nb",
]


@jit(nopython=True)
def rhs_ring_nb(t, theta, omega, k1, k2, r):
    """
    ODE RHS for coupled oscillators on a ring network.

    The coupling range is r, and interactions are pairwise and triadic.
    The coupling functions are:
    * sin(Oj - Oi)
    * sin(Oj + Ok - 2 Oi)

    Parameters
    ----------
    t : float
        Time (does not affect the result, there for consistency with integrators).
    theta : numpy ndarray
        Phases of the oscillators at time t.
    omega : float or array of floats
        Frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength
    k2 : float
        Triadic coupling strength
    r : int
        Pairwise and triplet nearest neighbour ranges

    Returns
    -------
    array
        Amount to add to the phases to update them after one integration step.

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

    return omega + (k1 / r) * pairwise + k2 / (r * (2 * r - 1)) * triplets
