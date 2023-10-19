import numpy as np
import pytest

import hypersync as hs


def test_generate_k_clusters():

    N = 10
    k = 1
    ps = [1]
    noise = 0
    state = hs.generate_k_clusters(N, k, ps, noise)

    assert isinstance(state, np.ndarray)
    assert len(state) == N
    assert len(set(state)) == 1

    # noise
    noise = 1e-2
    state = hs.generate_k_clusters(N, k, ps, noise)

    assert isinstance(state, np.ndarray)
    assert len(state) == N
    assert len(set(state)) > 1
    assert np.std(state) > 0
    assert np.isclose(np.std(state), noise, atol=1e-2)

    # k = 2
    N = 10
    k = 2
    ps = [0.3, 0.7]
    noise = 0
    state = hs.generate_k_clusters(N, k, ps, noise, seed=1)

    assert isinstance(state, np.ndarray)
    assert len(state) == N
    assert len(set(state)) == 2

    # random seed
    state1 = hs.generate_k_clusters(N, k, ps, seed=1)
    state2 = hs.generate_k_clusters(N, k, ps, seed=1)
    state3 = hs.generate_k_clusters(N, k, ps, seed=2)

    assert np.allclose(state1, state2)
    assert not np.allclose(state1, state3)

    # errors
    with pytest.raises(ValueError):
        hs.generate_k_clusters(10, 2, [0.5, 0.6])

    with pytest.raises(ValueError):
        hs.generate_k_clusters(10, 2, [0.5, 0.6, 0.4])
