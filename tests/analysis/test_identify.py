import numpy as np
import pytest

import hypersync as hs


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

    # 2-cluster state
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
    assert np.allclose(R1, 0.6, atol=1e-3)
    assert np.allclose(R2, 1, atol=1e-3)
