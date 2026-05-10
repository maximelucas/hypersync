"""
Sync state
==========

In a synchronized state all oscillators share the same phase.
The order parameter R1 = 1.
"""
import matplotlib.pyplot as plt

import hypersynchronization as hs

N = 20
theta = hs.generate_state(N, kind="sync", noise=0, seed=0)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
hs.plot_phases(theta, ax=axes[0])
hs.plot_phases_line(theta, ax=axes[1])
plt.tight_layout()
plt.show()
