import numpy as np
import pytest

import hypersync as hs


def test_generate_q_twisted_state():

    N = 20

    # shape
    theta = hs.generate_q_twisted_state(N, q=1, noise=0)
    assert theta.shape == (N,)

    # q=1: consecutive differences all equal 2π/N
    assert np.allclose(np.diff(theta), 2 * np.pi / N)

    # q=2: consecutive differences all equal 4π/N
    theta = hs.generate_q_twisted_state(N, q=2, noise=0)
    assert np.allclose(np.diff(theta), 4 * np.pi / N)

    # noise adds variability
    theta_clean = hs.generate_q_twisted_state(N, q=1, noise=0, seed=1)
    theta_noisy = hs.generate_q_twisted_state(N, q=1, noise=1e-1, seed=1)
    assert not np.allclose(theta_clean, theta_noisy)

    # reproducibility
    theta_a = hs.generate_q_twisted_state(N, q=1, seed=1)
    theta_b = hs.generate_q_twisted_state(N, q=1, seed=1)
    theta_c = hs.generate_q_twisted_state(N, q=1, seed=2)
    assert np.allclose(theta_a, theta_b)
    assert not np.allclose(theta_a, theta_c)


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

    # reproducibility
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


def test_generate_state():

    N = 20

    # shape for all kinds
    for kind in ["sync", "random", "splay"]:
        theta = hs.generate_state(N, kind=kind)
        assert theta.shape == (N,), kind

    theta = hs.generate_state(N, kind="k-cluster", k=2, ps=[0.5, 0.5])
    assert theta.shape == (N,)

    theta = hs.generate_state(N, kind="q-twisted", q=1)
    assert theta.shape == (N,)

    # random: values in [0, 2π]
    theta = hs.generate_state(N, kind="random", seed=1)
    assert np.all(theta >= 0)
    assert np.all(theta <= 2 * np.pi)

    # sync: all values close to the same phase
    theta = hs.generate_state(N, kind="sync", noise=1e-3, seed=1)
    assert np.std(theta) < 1e-2

    # splay (no noise): evenly spaced on [0, 2π]
    theta = hs.generate_state(N, kind="splay", noise=0)
    assert np.allclose(np.diff(theta), 2 * np.pi / N)

    # reproducibility
    theta_a = hs.generate_state(N, kind="random", seed=1)
    theta_b = hs.generate_state(N, kind="random", seed=1)
    theta_c = hs.generate_state(N, kind="random", seed=2)
    assert np.allclose(theta_a, theta_b)
    assert not np.allclose(theta_a, theta_c)

    # invalid kind
    with pytest.raises(ValueError):
        hs.generate_state(N, kind="invalid")
