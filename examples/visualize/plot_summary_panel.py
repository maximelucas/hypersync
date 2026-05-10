"""
Summary panel
=============

:func:`~hypersynchronization.plot_summary` produces a six-panel overview:
time series, order parameters, phase circle snapshots, and a ring plot —
all in one call.
"""
import matplotlib.pyplot as plt
import numpy as np
import xgi

import hypersynchronization as hs

N = 20
rng = np.random.default_rng(42)

H = xgi.complete_hypergraph(N, max_order=2)
links = H.edges.filterby("order", 1).members()
triangles = H.edges.filterby("order", 2).members()

thetas, times = hs.simulate_kuramoto(
    H,
    omega=rng.normal(0, 0.1, N),
    theta_0=hs.generate_state(N, kind="random", seed=42),
    t_end=30,
    dt=0.1,
    rhs=hs.rhs_23_sym,
    integrator="RK45",
    args=(2.0, 1.0, links, triangles),
)

fig = hs.plot_summary(thetas, times, H)
plt.show()
