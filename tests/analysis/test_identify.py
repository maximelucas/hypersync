import numpy as np
import pytest

import hypersync as hs


def test_identify_q_twisted():

    N = 20

    # q=1: phases wind once around the circle
    theta = hs.generate_q_twisted_state(N, q=1, noise=0, seed=1)
    q, is_twisted = hs.identify_q_twisted(theta)
    assert q == 1
    assert is_twisted

    # q=2: phases wind twice
    theta = hs.generate_q_twisted_state(N, q=2, noise=0, seed=1)
    q, is_twisted = hs.identify_q_twisted(theta)
    assert q == 2
    assert is_twisted

    # sync state is q=0 twisted
    theta = np.ones(N)
    q, is_twisted = hs.identify_q_twisted(theta)
    assert q == 0
    assert is_twisted

    # random state: not twisted
    theta = hs.generate_state(N, kind="random", seed=1)
    _, is_twisted = hs.identify_q_twisted(theta)
    assert not is_twisted


def test_identity_k_clusters():

    N = 10
    k = 1
    ps = [1]
    noise = 0
    state = hs.generate_k_clusters(N, k, ps, noise)

    is_1_cluster, sizes1 = hs.identify_k_clusters(state, 1)
    is_2_clusters, sizes2 = hs.identify_k_clusters(state, 2)
    is_3_clusters, sizes3 = hs.identify_k_clusters(state, 3)
    assert is_1_cluster
    assert not is_2_clusters
    assert not is_3_clusters

    # noise
    noise = 1e-1
    state = hs.generate_k_clusters(N, k, ps, noise, seed=2)

    is_1_cluster, sizes1 = hs.identify_k_clusters(state, 1)
    is_2_clusters, sizes2 = hs.identify_k_clusters(state, 2)
    is_3_clusters, sizes3 = hs.identify_k_clusters(state, 3)
    assert not is_1_cluster
    assert not is_2_clusters
    assert not is_3_clusters
    is_1_cluster, sizes1 = hs.identify_k_clusters(state, 1, atol=1e-1)
    assert is_1_cluster
    assert not is_2_clusters
    assert not is_3_clusters

    # k = 2
    N = 1000000
    k = 2
    ps = [0.3, 0.7]
    noise = 0
    state = hs.generate_k_clusters(N, k, ps, noise)

    is_1_cluster, sizes1 = hs.identify_k_clusters(state, 1)
    is_2_clusters, sizes2 = hs.identify_k_clusters(state, 2)
    is_3_clusters, sizes3 = hs.identify_k_clusters(state, 3)

    assert not is_1_cluster
    assert is_2_clusters
    assert np.allclose(sorted(sizes2), ps, atol=1e-2)
    assert not is_3_clusters


def test_identify_state():

    N = 20

    # sync (no noise) is identified as 0-twisted
    theta = np.ones(N)
    assert hs.identify_state(theta) == "0-twisted"

    # q-twisted states
    theta = hs.generate_q_twisted_state(N, q=1, noise=0, seed=1)
    assert hs.identify_state(theta) == "1-twisted"

    theta = hs.generate_q_twisted_state(N, q=2, noise=0, seed=1)
    assert hs.identify_state(theta) == "2-twisted"

    # splay: evenly spaced phases in non-index order
    theta = np.random.default_rng(42).permutation(
        np.linspace(0, 2 * np.pi, N, endpoint=False)
    )
    assert hs.identify_state(theta) == "splay"

    # 2-cluster
    theta = np.array([0.0] * 10 + [np.pi] * 10)
    assert hs.identify_state(theta) == "2-cluster"

    # 3-cluster
    theta = np.array([0.0] * 7 + [2 * np.pi / 3] * 7 + [4 * np.pi / 3] * 6)
    assert hs.identify_state(theta) == "3-cluster"

    # incoherent: random phases
    theta = hs.generate_state(N, kind="random", seed=42)
    assert hs.identify_state(theta) == "other"


def test_order_parameter():

    N = 100
    w = 0.1
    times = np.arange(0, 5, 1)
    n_times = len(times)

    # synthetic sync state rotating
    psi_0 = np.ones(N)
    psi = psi_0[:, None] + w * times

    R1 = hs.order_parameter(psi, order=1)
    R2 = hs.order_parameter(psi, order=2)

    assert R1.shape == (n_times,)
    assert R2.shape == (n_times,)
    assert np.allclose(R1, 1)
    assert np.allclose(R2, 1)

    # synthetic incoherent state rotating
    psi_0 = np.linspace(0, 2 * np.pi, endpoint=False, num=N)
    psi = psi_0[:, None] + w * times

    R1 = hs.order_parameter(psi, order=1)
    R2 = hs.order_parameter(psi, order=2)

    assert R1.shape == (n_times,)
    assert R2.shape == (n_times,)
    assert np.allclose(R1, 0)
    assert np.allclose(R2, 0)

    # 2-cluster state: R1 reflects imbalance, R2 ≈ 1
    N = 10
    k = 2
    ps = [0.3, 0.7]
    noise = 1e-2
    psi_0 = hs.generate_k_clusters(N, k, ps, noise, seed=1)
    psi = psi_0[:, None] + w * times

    R1 = hs.order_parameter(psi, order=1)
    R2 = hs.order_parameter(psi, order=2)

    assert R1.shape == (n_times,)
    assert R2.shape == (n_times,)
    assert np.allclose(R1, 0.4, atol=1e-3)
    assert np.allclose(R2, 1, atol=1e-3)

    # complex=True returns complex values
    Z = hs.order_parameter(psi_0, order=1, complex=True)
    assert np.iscomplex(Z)
    assert np.isclose(np.abs(Z), hs.order_parameter(psi_0, order=1))
