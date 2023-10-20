import numpy as np
import pytest

import hypersync as hs


def test_rhs_ring_nb():

    omega = 0
    k1 = 1
    k2 = 1

    # r = 1
    r = 1
    psi = np.array([0, np.pi / 2, 0, np.pi / 2])
    N = len(psi)
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, [2, -2, 2, -2])

    # r = 2
    r = 2
    psi = np.array([0, np.pi / 2, 0, np.pi / 2])
    N = len(psi)
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, [2.3333, -2.3333, 2.3333, -2.3333], atol=1e-3)

    # sync
    psi = np.ones(N)
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, 0)

    # 2-cluster
    psi = np.array([1, 1 + np.pi, 1, 1 + np.pi])
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, 0)

    # random
    psi = hs.generate_state(10, kind="random", seed=1)
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(
        psi_out,
        [
            -0.61466449,
            0.2096792,
            -0.94804897,
            0.22004142,
            -0.28502981,
            -0.10858078,
            0.18618909,
            0.17115625,
            0.52878709,
            0.08056231,
        ],
        atol=1e-5,
    )
