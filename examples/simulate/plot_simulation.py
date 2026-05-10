"""
Simulating Kuramoto oscillators
================================

Minimal end-to-end example: build a hypergraph, integrate the ODE from
random initial conditions, and visualize the result.
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

k1, k2 = 2.0, 1.0
omega = rng.normal(0, 0.1, N)
theta_0 = hs.generate_state(N, kind="random", seed=42)

thetas, times = hs.simulate_kuramoto(
    H,
    omega=omega,
    theta_0=theta_0,
    t_end=30,
    dt=0.1,
    rhs=hs.rhs_23_sym,
    integrator="RK45",
    args=(k1, k2, links, triangles),
)

fig, axs = hs.plot_sync(thetas, times)
plt.tight_layout()
plt.show()
