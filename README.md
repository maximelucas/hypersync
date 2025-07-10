# hypersync

hypersync provides code for the simulation, analysis, and visualization of oscillators with group (higher-order) interactions.


## Getting started

Here is a simple example to get started:

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import xgi

import hypersync as hs

sb.set_theme(style="ticks", context="notebook")

# structural parameters 
N = 20 # number of nodes
H = xgi.random_simplicial_complex(N, ps=[0.01, 0.015], seed=1) # hypergraph
links = H.edges.filterby("order", 1).members()
triangles = H.edges.filterby("order", 2).members()

# dynamical parameters
k1 = 5  # pairwise coupling strength
k2 = 0.1  # triplet coupling strength

# integration parameters
t_end = 300
dt = 0.01
integrator = "RK45"

# generate initial contitions
psi_init = hs.generate_state(N, kind="random")

# simulate system
thetas, times = hs.simulate_kuramoto(
    H,
    omega=np.random.normal(size=N),
    theta_0=psi_init,
    t_end=t_end,
    dt=dt,
    rhs=hs.rhs_23_sym, 
    integrator=integrator,
    args=(k1, k2, links, triangles), # arguments of the RHS function,
)


hs.plot_summary(thetas, times, H)

plt.show()
```

## Credits

HyperSync is makes use of XGI for higher-order interactions and Scipy's solve_ivp() for ODE integration.

Released under the 3-Clause BSD license.

Maxime Lucas: maxime.lucas.work [at] gmail.com