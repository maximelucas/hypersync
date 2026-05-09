"""
Right-hand-side functions of ODEs for simulating synchronisation
of oscillators with group interactions.
"""

from math import sin

import numpy as np
from numba import jit

__all__ = [
    "rhs_23_sym",
    "rhs_pairwise_a2a",
    "rhs_pairwise_meso",
    "rhs_pairwise_adj",
    "rhs_pairwise_a2a_harmonic",
    "rhs_triplet_sym_meso",
    "rhs_triplet_asym_meso",
    "rhs_23_sym_nb",
    "rhs_23_asym_nb",
    "rhs_ring_23_sym_nb",
    "rhs_ring_23_asym_nb",
]


def rhs_23_sym(t, theta, omega, k1, k2, links, triangles):
    """
    Right-hand side of the ODE for any hypergraph with links and triangles.

    Coupling functions: sin(theta_j - theta_i) and sin(theta_j + theta_k - 2*theta_i).
    Normalised by mean pairwise degree and mean triadic degree.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators at time t.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength.
    k2 : float
        Triadic coupling strength.
    links : list of tuples
        Pairwise edges, each as a tuple/list/set of two node indices.
    triangles : list of tuples
        Triadic edges, each as a tuple/list/set of three node indices.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)
    pairwise = np.zeros(N)
    triplet = np.zeros(N)

    for i, j in links:
        # sin(oj - oi)
        oi = theta[i]
        oj = theta[j]
        pairwise[i] += sin(oj - oi)
        pairwise[j] += sin(oi - oj)

    for i, j, k in triangles:
        # sin(oj + ok - 2*oi)
        oi = theta[i]
        oj = theta[j]
        ok = theta[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    k1_avg = len(links) / N * 2
    k2_avg = len(triangles) / N * 3
    g1 = (k1 / k1_avg) if k1_avg != 0 else 0
    g2 = (k2 / (k2_avg * 2)) if k2_avg != 0 else 0

    return omega + g1 * pairwise + g2 * triplet


def rhs_pairwise_a2a(t, theta, omega, k1):
    """
    Right-hand side of the ODE, all-to-all pairwise coupling.

    Coupling function: sin(theta_j - theta_i). This is the original Kuramoto
    model. Uses a vectorised implementation that avoids explicit matrix products.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Coupling strength.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.

    See Also
    --------
    rhs_pairwise_meso, rhs_pairwise_adj
    """
    N = len(theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    sum_cos = np.sum(cos_theta)
    sum_sin = np.sum(sin_theta)

    # oj - oi
    pairwise = -sum_cos * sin_theta + sum_sin * cos_theta

    return omega + (k1 / N) * pairwise


def rhs_pairwise_meso(t, theta, omega, k1, links):
    """
    Right-hand side of the ODE, pairwise coupling on an arbitrary network.

    Coupling function: sin(theta_j - theta_i). Loops over links.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Coupling strength.
    links : list of tuples
        Pairwise edges, each as a tuple/list/set of two node indices.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.

    See Also
    --------
    rhs_pairwise_adj
    """
    N = len(theta)
    pairwise = np.zeros(N)

    for i, j in links:
        # sin(oj - oi)
        oi = theta[i]
        oj = theta[j]
        pairwise[i] += sin(oj - oi)
        pairwise[j] += sin(oi - oj)

    return omega + (k1 / N) * pairwise


def rhs_pairwise_adj(t, theta, omega, k1, adj1):
    """
    Right-hand side of the ODE, pairwise coupling on an arbitrary network.

    Coupling function: sin(theta_j - theta_i). Uses matrix multiplication.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Coupling strength.
    adj1 : numpy.ndarray of shape (N, N)
        Adjacency matrix of order 1.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.

    See Also
    --------
    rhs_pairwise_meso
    """
    N = len(theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # oj - oi
    pairwise = adj1.dot(sin_theta) * cos_theta - adj1.dot(cos_theta) * sin_theta

    return omega + (k1 / N) * pairwise


def rhs_pairwise_a2a_harmonic(t, theta, omega, k1, k2):
    """
    Right-hand side of the ODE, all-to-all pairwise coupling with a harmonic.

    Coupling function: k1*sin(theta_j - theta_i) + k2*sin(2*theta_j - 2*theta_i).
    Uses a vectorised implementation that avoids explicit matrix products.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Coupling strength of the first harmonic.
    k2 : float
        Coupling strength of the second harmonic.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.

    See Also
    --------
    rhs_pairwise_a2a
    """
    N = len(theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    sum_cos = np.sum(cos_theta)
    sum_sin = np.sum(sin_theta)

    # oj - oi
    pairwise = -sum_cos * sin_theta + sum_sin * cos_theta

    sum_cos_sq = np.sum(cos_theta**2)
    sum_sin_sq = np.sum(sin_theta**2)

    # 2oj - 2oi
    pairwise_harmonic = 2 * (
        -cos_theta * sin_theta * sum_cos_sq
        + cos_theta**2 * np.sum(cos_theta * sin_theta)
        - sin_theta**2 * np.sum(cos_theta * sin_theta)
        + cos_theta * sin_theta * sum_sin_sq
    )

    return omega + (k1 / N) * pairwise + (k2 / N) * pairwise_harmonic


def rhs_triplet_sym_meso(t, theta, omega, k2, triangles):
    """
    Right-hand side of the ODE, symmetric triadic coupling on an arbitrary network.

    Coupling function: sin(theta_j + theta_k - 2*theta_i). Loops over triangles.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k2 : float
        Triadic coupling strength.
    triangles : list of tuples
        Triadic edges, each as a tuple/list/set of three node indices.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)
    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(oj + ok - 2*oi)
        oi = theta[i]
        oj = theta[j]
        ok = theta[k]
        triplet[i] += 2 * sin(oj + ok - 2 * oi)
        triplet[j] += 2 * sin(oi + ok - 2 * oj)
        triplet[k] += 2 * sin(oj + oi - 2 * ok)

    return omega + (k2 / N**2) * triplet


def rhs_triplet_asym_meso(t, theta, omega, k2, triangles):
    """
    Right-hand side of the ODE, asymmetric triadic coupling on an arbitrary network.

    Coupling function: sin(2*theta_j - theta_k - theta_i). Loops over triangles.

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k2 : float
        Triadic coupling strength.
    triangles : list of tuples
        Triadic edges, each as a tuple/list/set of three node indices.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)
    triplet = np.zeros(N)

    for i, j, k in triangles:
        # sin(2*oj - ok - oi)
        oi = theta[i]
        oj = theta[j]
        ok = theta[k]
        triplet[i] += sin(2 * oj - ok - oi) + sin(2 * ok - oj - oi)
        triplet[j] += sin(2 * ok - oi - oj) + sin(2 * oi - ok - oj)
        triplet[k] += sin(2 * oi - oj - ok) + sin(2 * oj - oi - ok)

    return omega + (k2 / N**2) * triplet


@jit(nopython=True)
def rhs_23_sym_nb(t, theta, omega, k1, k2):
    """
    ODE RHS for coupled oscillators on a complete network.

    Coupling functions: sin(theta_j - theta_i) and sin(theta_j + theta_k - 2*theta_i).

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators at time t.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength.
    k2 : float
        Triadic coupling strength.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    for ii in range(N):
        for jj in range(N):
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

            for kk in range(N):
                if jj < kk:
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    return omega + (k1 / N) * pairwise + k2 / (N**2) * triplets


@jit(nopython=True)
def rhs_23_asym_nb(t, theta, omega, k1, k2):
    """
    ODE RHS for coupled oscillators on a complete network.

    Coupling functions: sin(theta_j - theta_i) and sin(2*theta_j - theta_k - theta_i).

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators at time t.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength.
    k2 : float
        Triadic coupling strength.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    for ii in range(N):
        for jj in range(N):
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

            for kk in range(N):
                if jj == kk or ii == kk or ii == jj:
                    continue

                jjj = (ii + jj) % N
                kkk = (ii + kk) % N
                triplets[ii] += sin(-theta[kkk] + 2 * theta[jjj] - theta[ii])

    return omega + (k1 / N) * pairwise + k2 / (N**2) * triplets


@jit(nopython=True)
def rhs_ring_23_sym_nb(t, theta, omega, k1, k2, r):
    """
    ODE RHS for coupled oscillators on a ring network.

    Coupling functions: sin(theta_j - theta_i) and sin(theta_j + theta_k - 2*theta_i).

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators at time t.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength.
    k2 : float
        Triadic coupling strength.
    r : int
        Pairwise and triplet nearest-neighbour range.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    idx_2 = list(range(-r, 0)) + list(range(1, r + 1))
    idx_1 = range(-r, r + 1)

    for ii in range(N):
        for jj in idx_1:
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:
            for kk in idx_2:
                if jj < kk:
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    triplets[ii] += 2 * sin(theta[kkk] + theta[jjj] - 2 * theta[ii])

    return omega + (k1 / r) * pairwise + k2 / (r * (2 * r - 1)) * triplets


@jit(nopython=True)
def rhs_ring_23_asym_nb(t, theta, omega, k1, k2, r):
    """
    ODE RHS for coupled oscillators on a ring network.

    Coupling functions: sin(theta_j - theta_i) and sin(2*theta_j - theta_k - theta_i).

    Parameters
    ----------
    t : float
        Time (unused, present for compatibility with integrators).
    theta : numpy.ndarray
        Phases of the oscillators at time t.
    omega : float or numpy.ndarray
        Natural frequencies of the oscillators.
    k1 : float
        Pairwise coupling strength.
    k2 : float
        Triadic coupling strength.
    r : int
        Pairwise and triplet nearest-neighbour range.

    Returns
    -------
    dtheta : numpy.ndarray of shape (N,)
        Rate of change of phases.
    """
    N = len(theta)

    pairwise = np.zeros(N)
    triplets = np.zeros(N)

    idx_2 = list(range(-r, 0)) + list(range(1, r + 1))
    idx_1 = range(-r, r + 1)

    for ii in range(N):
        for jj in idx_1:
            jjj = (ii + jj) % N
            pairwise[ii] += sin(theta[jjj] - theta[ii])

        for jj in idx_2:
            for kk in idx_2:
                if jj != kk:
                    jjj = (ii + jj) % N
                    kkk = (ii + kk) % N
                    triplets[ii] += sin(-theta[kkk] + 2 * theta[jjj] - theta[ii])

    return omega + (k1 / r) * pairwise + k2 / (r * (2 * r - 1)) * triplets
