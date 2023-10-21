import numpy as np
import pytest
import xgi

import hypersync as hs

def test_rhs_pairwise_a2a(state2): 

    omega = 1
    k1 = 1
    N = 10

    psi_out = hs.rhs_pairwise_a2a(0, state2, omega, k1)

    # coherence with other rhs
    H = xgi.complete_hypergraph(N, max_order=2)
    adj1 = xgi.adjacency_matrix(H, order=1)
    psi_out2 = hs.rhs_pairwise_adj(0, state2, omega, k1, adj1)
    assert np.allclose(psi_out, psi_out2)


def test_rhs_pairwise_meso(hypergraph0, state1):

    omega = 0
    k1 = 1
    H = hypergraph0
    H = xgi.convert_labels_to_integers(H, "label")

    links = H.edges.filterby("order", 1).members()

    psi = np.array([0, np.pi / 2, np.pi / 2, 0, np.pi])
    psi_out = hs.rhs_pairwise_meso(0, psi, omega, k1, links)
    assert np.allclose(psi_out, [0.4, -0.2, -0.2, 0, 0])

    # random ic
    psi_out = hs.rhs_pairwise_meso(0, state1, omega, k1, links)
    assert np.allclose(
        psi_out, [0.33054527, -0.15332373, -0.00983145, -0.32393079, 0.15654069]
    )

    # consistency with other rhs
    adj1 = xgi.adjacency_matrix(H, order=1)
    psi_out2 = hs.rhs_pairwise_adj(0, state1, omega, k1, adj1)
    assert np.allclose(psi_out, psi_out2)


def test_rhs_pairwise_adj(hypergraph0, state1):

    omega = 0
    k1 = 1
    H = hypergraph0
    H = xgi.convert_labels_to_integers(H, "label")

    adj1 = xgi.adjacency_matrix(H, order=1)
    psi_out = hs.rhs_pairwise_adj(0, state1, omega, k1, adj1)
    assert np.allclose(
        psi_out, [0.33054527, -0.15332373, -0.00983145, -0.32393079, 0.15654069]
    )

    # consistency with other rhs
    links = H.edges.filterby("order", 1).members()
    psi_out2 = hs.rhs_pairwise_meso(0, state1, omega, k1, links)
    assert np.allclose(psi_out, psi_out2)


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
