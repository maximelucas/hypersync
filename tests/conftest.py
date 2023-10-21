import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xgi

import hypersync as hs


@pytest.fixture
def state1():
    N = 5
    return hs.generate_state(N, ic="random", seed=1)


@pytest.fixture
def edgelist1():
    return [{1, 2}, {2, 3}, {3, 1}, {1, 4}, {4, 5}, {1, 2, 3}, {3, 4, 5}]


@pytest.fixture
def hypergraph0(edgelist1):
    H = xgi.Hypergraph(edgelist1)
    return H


@pytest.fixture
def hypergraph1():
    H = xgi.Hypergraph()
    H.add_nodes_from(["a", "b", "c"])
    H.add_edges_from({"e1": ["a", "b"], "e2": ["a", "b", "c"], "e3": ["c"]})
    return H


@pytest.fixture
def hypergraph2():
    H = xgi.Hypergraph()
    H.add_nodes_from(["b", "c", 0])
    H.add_edges_from({"e1": [0, "b"], "e2": [0, "c"], "e3": [0, "b", "c"]})
    return H


@pytest.fixture
def simplicialcomplex1():
    S = xgi.SimplicialComplex()
    S.add_nodes_from(["b", "c", 0])
    S.add_edges_from(
        {"e1": [0, "b"], "e2": [0, "c"], "e3": [0, "b", "c"], "e4": ["b", "c"]}
    )
    return S


@pytest.fixture
def dihypergraph1():
    H = xgi.DiHypergraph()
    H.add_nodes_from(["a", "b", "c", "d"])
    H.add_edges_from(
        {"e1": [{"a", "b"}, {"c"}], "e2": [{"b"}, {"c", "d"}], "e3": [{"b"}, {"c"}]}
    )
    return H
