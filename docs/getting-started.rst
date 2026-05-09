Getting started
===============

Here is a minimal example simulating Kuramoto oscillators with pairwise
and triplet interactions on a random hypergraph:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import xgi

   import hypersynchronization as hs

   N = 20
   H = xgi.random_simplicial_complex(N, ps=[0.01, 0.015], seed=1)
   links = H.edges.filterby("order", 1).members()
   triangles = H.edges.filterby("order", 2).members()

   k1 = 5
   k2 = 0.1

   theta_0 = hs.generate_state(N, kind="random")

   thetas, times = hs.simulate_kuramoto(
       H,
       omega=np.random.normal(size=N),
       theta_0=theta_0,
       t_end=300,
       dt=0.01,
       rhs=hs.rhs_23_sym,
       integrator="RK45",
       args=(k1, k2, links, triangles),
   )

   hs.plot_summary(thetas, times, H)
   plt.show()
