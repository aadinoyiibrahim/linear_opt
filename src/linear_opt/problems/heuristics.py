"""Construction and improvement heuristics used as MIP starts.

A good incumbent from the start lets branch-and-bound prune immediately; these
classical heuristics are cheap and usually within a few percent of optimal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from linear_opt.core.ir import FloatArray


def nearest_neighbour(dist: FloatArray, start: int = 0) -> list[int]:
    """Greedy tour: always visit the closest unvisited node."""
    n = dist.shape[0]
    tour = [start]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False
    for _ in range(n - 1):
        row = np.where(unvisited, dist[tour[-1]], np.inf)
        nxt = int(np.argmin(row))
        tour.append(nxt)
        unvisited[nxt] = False
    return tour


def two_opt(dist: FloatArray, tour: Sequence[int], max_rounds: int = 1000) -> list[int]:
    """Best-improvement 2-opt: reverse segments while that shortens the tour."""
    t = np.asarray(tour, dtype=np.int64)
    n = len(t)
    if n < 4:
        return t.tolist()
    for _ in range(max_rounds):
        best_delta, best = -1e-9, None
        for i in range(n - 2):
            a, b = t[i], t[i + 1]
            js = np.arange(i + 2, n if i > 0 else n - 1)
            if js.size == 0:
                continue
            c, d = t[js], t[(js + 1) % n]
            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            k = int(np.argmin(delta))
            if delta[k] < best_delta:
                best_delta, best = float(delta[k]), (i, int(js[k]))
        if best is None:
            break
        i, j = best
        t[i + 1 : j + 1] = t[i + 1 : j + 1][::-1]
    return t.tolist()


def clarke_wright(
    dist: FloatArray, demand: FloatArray, capacity: float, depot: int = 0
) -> list[list[int]]:
    """Parallel Clarke-Wright savings: merge routes in order of ``d0i + d0j - dij``."""
    n = dist.shape[0]
    customers = [v for v in range(n) if v != depot]
    routes: dict[int, list[int]] = {v: [v] for v in customers}
    route_of = {v: v for v in customers}
    load = {v: float(demand[v]) for v in customers}
    savings = sorted(
        (
            (dist[depot, i] + dist[depot, j] - dist[i, j], i, j)
            for a, i in enumerate(customers)
            for j in customers[a + 1 :]
        ),
        reverse=True,
    )
    for saving, i, j in savings:
        if saving <= 0:
            break
        ri, rj = route_of[i], route_of[j]
        if ri == rj or load[ri] + load[rj] > capacity + 1e-9:
            continue
        a, b = routes[ri], routes[rj]
        # i and j must be route ends; orient so that ... i] + [j ...
        if a[-1] != i:
            if a[0] != i:
                continue
            a.reverse()
        if b[0] != j:
            if b[-1] != j:
                continue
            b.reverse()
        routes[ri] = a + b
        load[ri] += load.pop(rj)
        for v in routes.pop(rj):
            route_of[v] = ri
    return list(routes.values())
