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
    state = hs.generate_k_clusters(N, k, ps, noise)

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
    assert np.allclose(sizes2, ps, atol=1e-3)
    assert not is_3_clusters
