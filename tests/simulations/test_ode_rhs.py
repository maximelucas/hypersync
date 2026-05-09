import numpy as np
import xgi

import hypersync as hs


def test_rhs_23_sym(hypergraph0, state1):

    omega = 0
    k1 = 1
    k2 = 1
    H = hypergraph0
    H = xgi.convert_labels_to_integers(H, "label")
    links = H.edges.filterby("order", 1).members()
    triangles = H.edges.filterby("order", 2).members()

    # sync state: pairwise and triplet terms vanish
    theta_sync = np.ones(H.num_nodes)
    psi_out = hs.rhs_23_sym(0, theta_sync, omega, k1, k2, links, triangles)
    assert np.allclose(psi_out, 0)

    # output shape
    psi_out = hs.rhs_23_sym(0, state1, omega, k1, k2, links, triangles)
    assert psi_out.shape == (H.num_nodes,)

    # random ic against reference values
    assert np.allclose(
        psi_out,
        [0.37129268, -0.55189841, 0.49544906, -0.11487624, -0.21826764],
        atol=1e-6,
    )

    # omega shifts result uniformly
    omega_arr = np.ones(H.num_nodes) * 2.5
    psi_out_omega = hs.rhs_23_sym(0, state1, omega_arr, k1, k2, links, triangles)
    assert np.allclose(psi_out_omega, psi_out + 2.5)

    # k1=0: pairwise term vanishes, only triplet remains
    psi_k1 = hs.rhs_23_sym(
        0, state1, omega, k1=0, k2=1, links=links, triangles=triangles
    )
    psi_k2 = hs.rhs_23_sym(
        0, state1, omega, k1=1, k2=0, links=links, triangles=triangles
    )
    psi_both = hs.rhs_23_sym(
        0, state1, omega, k1=1, k2=1, links=links, triangles=triangles
    )
    assert np.allclose(psi_k1 + psi_k2, psi_both)

    # empty hypergraph: no coupling, returns omega
    psi_out = hs.rhs_23_sym(0, state1, omega, k1, k2, links=[], triangles=[])
    assert np.allclose(psi_out, omega)


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
        psi_out, [0.00473607, 0.11240593, -0.03982561, 0.07420681, -0.1515232]
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
        psi_out, [0.00473607, 0.11240593, -0.03982561, 0.07420681, -0.1515232]
    )

    # consistency with other rhs
    links = H.edges.filterby("order", 1).members()
    psi_out2 = hs.rhs_pairwise_meso(0, state1, omega, k1, links)
    assert np.allclose(psi_out, psi_out2)


def test_rhs_ring_23_asym_nb():

    omega = 0
    k1 = 1
    k2 = 1

    # r = 1
    r = 1
    psi = np.array([0, np.pi / 2, 0, np.pi / 2])
    N = len(psi)
    psi_out = hs.rhs_ring_23_sym_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, [2, -2, 2, -2])

    # r = 2
    r = 2
    psi = np.array([0, np.pi / 2, 0, np.pi / 2])
    N = len(psi)
    psi_out = hs.rhs_ring_23_sym_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, [2.3333, -2.3333, 2.3333, -2.3333], atol=1e-3)

    # sync
    psi = np.ones(N)
    psi_out = hs.rhs_ring_23_sym_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, 0)

    # 2-cluster
    psi = np.array([1, 1 + np.pi, 1, 1 + np.pi])
    psi_out = hs.rhs_ring_23_sym_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(psi_out, 0)

    # random
    psi = hs.generate_state(10, kind="random", seed=1)
    psi_out = hs.rhs_ring_23_sym_nb(0, psi, omega, k1, k2, r)
    assert np.allclose(
        psi_out,
        [
            -0.14940351,
            0.59211323,
            0.05487797,
            0.85278216,
            -0.61954368,
            0.13942231,
            0.5618456,
            0.12551108,
            0.41752752,
            0.27133915,
        ],
        atol=1e-5,
    )
