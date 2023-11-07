"""
Righ-hand-side functions of ODEs for simulating synchronisation 
of oscillators with group interactions
"""

from math import sin

import numpy as np
import xgi
from numba import jit

__all__ = [
    "rhs_pairwise_a2a",
    "rhs_pairwise_meso",
    "rhs_pairwise_adj",
    "rhs_triplet_sym_meso",
    "rhs_triplet_asym_meso",
    "rhs_23_sym_nb",
    "rhs_23_asym_nb",
    "rhs_ring_nb",
]


def rhs_pairwise_a2a(t, psi, omega, k1):
    """Right-hand side of the ODE, all-to-all pairwise coupling.

    Coupling function: sin(oj - oi). This is the original Kuramoto
    model. This version uses an optimisation avoiding matrix products.

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    omega : float or array of floats
        Natural frequencies of the oscillators
    k1 : float
        Coupling strength

    Returns
    -------
    array of floats of length N

    See also
    --------
    rhs_pairwise_meso
    rhs_pairwise_adj
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    sum_cos_psi = np.sum(cos_psi)
    sum_sin_psi = np.sum(sin_psi)

    # oj - oi
    pairwise = -sum_cos_psi * sin_psi + sum_sin_psi * cos_psi

    return omega + (k1 / N) * pairwise


def rhs_pairwise_meso(t, psi, omega, k1, links):
    """Right-hand side of the ODE, only pairwise coupling.

    Coupling function: sin(oj - oi). This is the usual Kuramoto
    model on complex networks. This version of the function loops
    over links.

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    omega : float or array of floats
        Natural frequencies of the oscillators
    k1 : float
        Coupling strength
    links : list of tuples
        List of pairwise links, in the form of list/tuple/set
        of two elements.

    Returns
    -------
    array of floats of length N

    See also
    --------
    rhs_pairwise_adj

    """

    N = len(psi)
    pairwise = np.zeros(N)

    for i, j in links:
        # sin(oj - oi)
        oi = psi[i]
        oj = psi[j]
        pairwise[i] += sin(oj - oi)
        pairwise[j] += sin(oi - oj)

    return omega + (k1 / N) * pairwise


def rhs_pairwise_adj(t, psi, omega, k1, adj1):
    """Right-hand side of the ODE, only pairwise coupling.

    Coupling function: sin(oj - oi). This is the usual Kuramoto
    model on complex networks. This version of the function uses
    matrix multiplication.

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    omega : float or array of floats
        Natural frequencies of the oscillators
    k1 : float
        Coupling strength
    adj1 : ndarray
        Adjacency matrix (of order 1) of shape (N, N).

    Returns
    -------
    array of floats of length N

    See also
    --------
    rhs_pairwise_adj
    """

    N = len(psi)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    pairwise = adj1.dot(sin_psi) * cos_psi - adj1.dot(cos_psi) * sin_psi

    return omega + (k1 / N) * pairwise


def rhs_triplet_sym_meso(t, psi, omega, k2, triangles):
    """Right-hand side of the ODE, only triadic coupling.

    Coupling function: sin(oj + ok - 2oi). This version of the function loops
    over triangles.

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    omega : float or array of floats
        Natural frequencies of the oscillators
    k2 : float
        Coupling strength
    triangles : list of tuples
        List of triadic edges, in the form of list/tuple/set
        of three elements.

    Returns
    -------
    array of floats of length N

    """

    N = len(psi)
    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(oj - ok - 2 oi)
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    return omega + (k2 / N**2) * triplet


def rhs_triplet_asym_meso(t, psi, omega, k2, triangles):
    """Right-hand side of the ODE, only triadic coupling.

    Coupling function: sin(2oj - ok + 2oi). This version of the function loops
    over triangles.

    Parameters
    ----------
    t: float
        Time
    psi: array of float
        Phases to integrate
    omega : float or array of floats
        Natural frequencies of the oscillators
    k2 : float
        Coupling strength
    triangles : list of tuples
        List of triadic edges, in the form of list/tuple/set
        of three elements.

    Returns
    -------
    array of floats of length N

    """

    N = len(psi)
    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(oj - ok - 2 oi)
        oi = psi[i]
        oj = psi[j]
        ok = psi[k]
        triplet[i] += sin(2 * oj - ok - oi) + sin(2 * ok - oj - oi)
        triplet[j] += sin(2 * ok - oi - oj) + sin(2 * oi - ok - oj)
        triplet[k] += sin(2 * oi - oj - ok) + sin(2 * oj - oi - ok)

    return omega + (k2 / N**2) * triplet


@jit(nopython=True)
def rhs_23_sym_nb(t, theta, omega, k1, k2):
    """
    ODE RHS for coupled oscillators on a complete network.

    The coupling range is r, and interactions are pairwise and triadic.
    The coupling functions are:
    * sin(oj - oi)
    * sin(oj + ok - 2oi)

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

    Returns
    -------
    array
        Amount to add to the phases to update them after one integration step.

    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    for ii in range(N):
        for jj in range(N):  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

            for kk in range(N):
                if jj < kk:  # because coupling function is symmetric in j and k
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    # x2 to count triangles in both directions
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    return omega + (k1 / N) * pairwise + k2 / (N**2) * triplets


@jit(nopython=True)
def rhs_23_asym_nb(t, theta, omega, k1, k2):
    """
    ODE RHS for coupled oscillators on a complete network.

    The coupling range is r, and interactions are pairwise and triadic.
    The coupling functions are:
    * sin(oj - oi)
    * sin(2oj - ok - oi)

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

    Returns
    -------
    array
        Amount to add to the phases to update them after one integration step.

    """

    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    for ii in range(N):
        for jj in range(N):  # pairwise
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

            for kk in range(N):
                jjj = (ii + jj) % N
                kkk = (ii + kk) % N
                triplets[ii] += sin(-theta[kkk] + 2 * theta[jjj] - theta[ii])

    return omega + (k1 / N) * pairwise + k2 / (N**2) * triplets


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
