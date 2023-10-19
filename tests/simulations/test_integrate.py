import numpy as np
import pytest

import hypersync as hs


def test_rhs_ring_nb():

    omega = 0
    k1 = 1
    k2 = 1
    r = 2

    psi = np.array([0, np.pi, 0, np.pi])
    psi_out = hs.rhs_ring_nb(0, psi, omega, k1, k2, r)
