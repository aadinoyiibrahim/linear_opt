from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linear_opt.core.graph import EdgeIndex, components, global_min_cut, mask, min_cut_phases


def test_edge_index_basics() -> None:
    e = EdgeIndex(5)
    assert e.m == 10
    assert e.id(3, 1) == e.id(1, 3)
    inc = e.incidence().toarray()
    assert inc.sum(axis=1).tolist() == [4] * 5  # complete graph: degree n-1
    s = mask(5, [0, 1])
    assert len(e.delta(s)) == 2 * 3 and len(e.inside(s)) == 1
    x = np.arange(e.m, dtype=float)
    np.testing.assert_array_equal(e.weights(e.to_matrix(x)), x)
    with pytest.raises(ValueError, match="self-loops"):
        e.id(2, 2)
    with pytest.raises(ValueError, match="two nodes"):
        EdgeIndex(1)


def test_components() -> None:
    w = np.zeros((5, 5))
    w[0, 1] = w[1, 0] = 1.0
    w[2, 3] = w[3, 2] = 0.5
    w[3, 4] = w[4, 3] = 1e-9  # below eps: not an edge
    comps = sorted(c.tolist() for c in components(w))
    assert comps == [[0, 1], [2, 3], [4]]


def _brute_force(w: np.ndarray) -> float:
    n = w.shape[0]
    return float(
        min(
            w[np.ix_(s := mask(n, list(c)), ~s)].sum()
            for r in range(1, n)
            for c in itertools.combinations(range(n), r)
        )
    )


@settings(max_examples=60, deadline=None)
@given(n=st.integers(2, 7), seed=st.integers(0, 10**6), density=st.floats(0.2, 1.0))
def test_stoer_wagner_matches_brute_force(n: int, seed: int, density: float) -> None:
    rng = np.random.default_rng(seed)
    w = np.triu(rng.random((n, n)) * (rng.random((n, n)) < density), 1)
    w = w + w.T
    value, side = global_min_cut(w)
    s = mask(n, side)
    assert value == pytest.approx(_brute_force(w))
    assert w[np.ix_(s, ~s)].sum() == pytest.approx(value)  # the side realises the value
    assert 0 < s.sum() < n
    assert len(min_cut_phases(w)) == n - 1
