"""
Splay state
===========

In a splay state phases are evenly distributed on the circle.
The order parameter R1 = 0. Phases increase linearly with node index.
"""
import matplotlib.pyplot as plt

import hypersynchronization as hs

N = 20
theta = hs.generate_state(N, kind="splay", noise=0)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
hs.plot_phases(theta, ax=axes[0])
hs.plot_phases_line(theta, ax=axes[1])
plt.tight_layout()
plt.show()
