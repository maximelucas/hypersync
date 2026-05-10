"""
Two-cluster state
=================

In a 2-cluster state oscillators split into two groups separated by π.
The order parameter R1 reflects the imbalance between cluster sizes,
and R2 = 1.
"""
import matplotlib.pyplot as plt

import hypersynchronization as hs

N = 20
theta = hs.generate_k_clusters(N, k=2, ps=[0.5, 0.5], noise=0, seed=0)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
hs.plot_phases(theta, ax=axes[0])
hs.plot_phases_line(theta, ax=axes[1])
plt.tight_layout()
plt.show()
