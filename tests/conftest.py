import numpy as np
import pytest
import xgi

import hypersync as hs


@pytest.fixture
def state1():
    N = 5
    return hs.generate_state(N, kind="random", seed=1)


@pytest.fixture
def state2():
    N = 10
    return hs.generate_state(N, kind="random", seed=1)


@pytest.fixture
def edgelist1():
    return [{1, 2}, {2, 3}, {3, 1}, {1, 4}, {4, 5}, {1, 2, 3}, {3, 4, 5}]


@pytest.fixture
def hypergraph0(edgelist1):
    return xgi.Hypergraph(edgelist1)


@pytest.fixture(scope="module")
def hypergraph_complete():
    """Complete hypergraph (all-to-all) with links and triangles, N=50."""
    return xgi.complete_hypergraph(50, max_order=2)


@pytest.fixture(scope="module")
def hypergraph_sparse():
    """Sparse random hypergraph, N=50."""
    return xgi.random_hypergraph(50, ps=[0.05, 0.02], seed=42)


@pytest.fixture(scope="module")
def hypergraph_dense():
    """Dense random hypergraph, N=50."""
    return xgi.random_hypergraph(50, ps=[0.5, 0.3], seed=42)
