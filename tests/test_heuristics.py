from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from linear_opt.data.tsplib import tour_length
from linear_opt.problems.cvrp import CVRPInstance
from linear_opt.problems.heuristics import clarke_wright, nearest_neighbour, two_opt
from linear_opt.problems.tsp import TSPInstance


@settings(max_examples=30, deadline=None)
@given(n=st.integers(3, 25), seed=st.integers(0, 10**6))
def test_nearest_neighbour_and_two_opt(n: int, seed: int) -> None:
    d = TSPInstance.random(n, seed=seed).dist
    nn = nearest_neighbour(d)
    assert sorted(nn) == list(range(n)) and nn[0] == 0
    improved = two_opt(d, nn)
    assert sorted(improved) == list(range(n))
    assert tour_length(d, improved) <= tour_length(d, nn) + 1e-9


@settings(max_examples=30, deadline=None)
@given(n=st.integers(1, 20), seed=st.integers(0, 10**6), cap=st.sampled_from([30.0, 60.0, 100.0]))
def test_clarke_wright_is_feasible(n: int, seed: int, cap: float) -> None:
    inst = CVRPInstance.random(n, seed=seed, capacity=cap)
    routes = clarke_wright(inst.dist, inst.demand, inst.capacity)
    assert sorted(v for r in routes for v in r) == list(range(1, n + 1))
    assert all(inst.demand[r].sum() <= cap + 1e-9 for r in routes)
    singletons = sum(inst.route_cost([v]) for v in range(1, n + 1))
    assert sum(inst.route_cost(r) for r in routes) <= singletons + 1e-9  # savings never hurt


def test_two_opt_untangles_a_crossing() -> None:
    pts = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)  # 0-1-2-3 crosses itself
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    assert tour_length(d, two_opt(d, [0, 1, 2, 3])) == 4.0
