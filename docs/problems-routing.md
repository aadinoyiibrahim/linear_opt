# Routing: TSP and CVRP

## Travelling salesman

| formulation | size | LP bound |
|---|---|---|
| DFJ + lazy subtour cuts | $n(n-1)/2$ binaries, $n$ rows + cuts | Held–Karp (strongest) |
| single-commodity flow | $2n^2$ variables, $O(n^2)$ rows | intermediate |
| MTZ | $n^2 + n$ variables, $O(n^2)$ rows | weakest |

Padberg & Sung (1991) show $z_{LP}^{MTZ} \le z_{LP}^{SCF} \le z_{LP}^{DFJ}$; the
test suite checks this ordering on random instances, using the same separation
routine to compute the DFJ bound by row generation.

**Separation.** For an integer point, the connected components of the support
graph are the subtours; each yields $x(\delta(S)) \ge 2$. For a fractional point
that is connected, the global minimum cut (Stoer–Wagner) is the most violated
subtour cut; every phase cut below 2 is added.

## Capacitated vehicle routing

Two-index formulation with rounded capacity inequalities
$x(\delta(S)) \ge 2\lceil d(S)/Q \rceil$. Exact separation is NP-hard; on integer
points, however, the customer components after removing the depot are exactly
the routes and subtours, so checking them is exact. Fractional points use the
component heuristic at several support thresholds.

## Heuristics as MIP starts

Nearest neighbour + 2-opt (TSP) and Clarke–Wright savings (CVRP) provide
incumbents before the first node; tests check that each start is feasible for
every formulation's constraint matrix.

::: linear_opt.problems.tsp

::: linear_opt.problems.cvrp

::: linear_opt.problems.heuristics

::: linear_opt.core.graph
