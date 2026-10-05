# Capacitated facility location

$$
\begin{aligned}
\min\; & \sum_i f_i y_i + \sum_{i,j} c_{ij} x_{ij} \\
\text{s.t.}\; & \textstyle\sum_i x_{ij} = 1 && \forall j \\
& \textstyle\sum_j d_j x_{ij} \le s_i y_i && \forall i \\
& x_{ij} \le y_i && \forall i, j \quad (\text{strong only}) \\
& \textstyle\sum_i s_i y_i \ge \sum_j d_j && (\text{optional cover cut}) \\
& x_{ij} \in [0, 1],\; y_i \in \{0, 1\}
\end{aligned}
$$

OR-Library's ``cap`` instances allow split demand; ``c_ij`` is the cost of
serving *all* of customer ``j`` from ``i``. Their published optima (``capopt.txt``)
are used as test oracles.

**Why the strong formulation is tighter.** In the weak model, a fractional
$y_i = \sum_j d_j x_{ij} / s_i$ suffices, so the LP pays only a fraction of
each opening cost. The disaggregated rows force $y_i \ge \max_j x_{ij}$.
Every LP point of the strong model is feasible for the weak one, so
$z^{LP}_{strong} \ge z^{LP}_{weak}$; the test suite checks this property on
random instances.

**What the tests check.** Agreement between backends; the cap41 published
optimum; strong/weak equality of the integer optimum and dominance of the LP
bound; single-sourcing cost ≥ split cost; feasibility of the greedy start; and an
independent re-check of every solution against the raw data.

::: linear_opt.problems.facility

::: linear_opt.core.study
