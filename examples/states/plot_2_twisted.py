"""
Two-twisted state
=================

In a q-twisted state the phases wind q times around the circle with
node index. For q=2 the phase-vs-index plot shows two full cycles,
distinguishing it from the splay state (q=1).
"""
import matplotlib.pyplot as plt

import hypersynchronization as hs

N = 20
theta = hs.generate_q_twisted_state(N, q=2, noise=0, seed=0)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
hs.plot_phases(theta, ax=axes[0])
hs.plot_phases_line(theta, ax=axes[1])
plt.tight_layout()
plt.show()
