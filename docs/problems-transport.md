# Transportation / optimal transport

$$
\min_{x \ge 0} \sum_{i,j} c_{ij} x_{ij}
\quad\text{s.t.}\quad
\sum_j x_{ij} \le s_i \;\; (u_i \le 0), \qquad
\sum_i x_{ij} \ge d_j \;\; (v_j \ge 0).
$$

With $\sum_i s_i = \sum_j d_j$ and equality rows this is the Kantorovich problem
of discrete optimal transport.

**Duals.** $v_j$ is the marginal cost of one more unit at customer $j$;
$-u_i$ is what one more unit of capacity at depot $i$ saves. A route is used only
if its reduced cost $c_{ij} - u_i - v_j$ is zero (complementary slackness).

**How the result is checked.** The tests do not trust the solver:

* `validate()` recomputes capacities, demands and cost from the raw data;
* the dual objective $b^\top y$ must equal the primal objective (strong duality);
* every dual has the sign its row sense requires;
* used routes have zero reduced cost;
* property tests: adding $k$ to all costs into sink $j$ raises the optimum by
  exactly $k\,d_j$; scaling all costs by $\alpha$ scales the optimum by $\alpha$.

::: linear_opt.problems.transport
