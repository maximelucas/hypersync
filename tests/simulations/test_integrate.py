import numpy as np
import pytest
import xgi

import hypersync as hs


def test_simulate_kuramoto():

    # test ring ode (structure is in the ode)

    N = 100
    H = xgi.trivial_hypergraph(N)

    k1 = 1
    k2 = 1
    r = 2
    args = (k1, k2, r)

    thetas_out, times_out = hs.simulate_kuramoto(H, args=args)

    times = np.arange(0, 100 + 0.01 / 2, 0.01)
    n_t = len(times)

    assert np.allclose(times, times_out)
    assert thetas_out.shape == (N, n_t)

    # set integration times

    t_end = 30
    dt = 0.1

    thetas_out, times_out = hs.simulate_kuramoto(H, args=args, t_end=t_end, dt=dt)

    times = np.arange(0, t_end + dt / 2, dt)
    n_t = len(times)

    assert np.allclose(times, times_out)
    assert thetas_out.shape == (N, n_t)

    # set initial condition

    psi = np.ones(N)
    thetas_out, times_out = hs.simulate_kuramoto(H, theta_0=psi, args=args)
    assert np.allclose(thetas_out[:, 0], psi)

    # set integrator
    t_end = 100
    dt = 0.1
    thetas_out, times_out = hs.simulate_kuramoto(
        H, integrator="RK45", args=args, t_end=t_end, dt=dt
    )

    n_t = len(times_out)

    assert n_t == t_end / dt + 1  # find case where it's smaller
    assert thetas_out.shape == (N, n_t)

    # t_eval
    t_end = 100
    dt = 0.1
    thetas_out, times_out = hs.simulate_kuramoto(
        H, integrator="RK45", args=args, t_eval=True, t_end=t_end, dt=dt
    )

    n_t = len(times_out)

    assert n_t == t_end / dt + 1  # find case where it's smaller
    assert thetas_out.shape == (N, n_t)

    # errors
    with pytest.raises(ValueError):
        hs.simulate_kuramoto(H, integrator="RK46", args=args)
    with pytest.raises(TypeError):
        hs.simulate_kuramoto(H)
