# Job-shop scheduling

## Disjunctive formulation (Manne 1960)

$$
\begin{aligned}
\min\; & C_{\max} \\
\text{s.t.}\; & s_{o+1} \ge s_o + p_o && \text{(job order)} \\
& C_{\max} \ge s_{\text{last}(j)} + p_{\text{last}(j)} \\
& s_a + p_a \le s_b + M_{ab}(1 - y_{ab}),\quad s_b + p_b \le s_a + M_{ba}\, y_{ab} && \mu_a = \mu_b \\
& r_o \le s_o \le H - q_o - p_o, \quad y_{ab} \in \{0,1\}
\end{aligned}
$$

with heads $r_o$ (work before $o$ in its job), tails $q_o$ (work after $o$) and a
horizon $H$ (an upper bound on the optimum). The pair-specific
$M_{ab} = \bar s_a + p_a - \underline s_b$ is the smallest valid value given the
start windows; `big_m = "tight"` takes $H$ from a Giffler–Thompson schedule,
`"naive"` uses $H = \sum_o p_o$.

## Time-indexed formulation (Bowman 1959)

$x_{o,t} = 1$ if $o$ starts at $t$. Machine capacity is
$\sum_{o:\mu_o=m}\sum_{\tau=t-p_o+1}^{t} x_{o,\tau} \le 1$ for every machine and
period, and precedence uses the **strong (disaggregated)** form

$$
\sum_{\tau \le t} x_{o+1,\tau} \le \sum_{\tau \le t - p_o} x_{o,\tau} \qquad \forall t,
$$

which is much tighter than $\sum_t t\,x_{o+1,t} \ge \sum_t t\,x_{o,t} + p_o$.

## Post-processing

Solutions are converted to semi-active form (left shift), which keeps the
machine sequences and cannot increase the makespan. The critical path is then a
chain of operations from time 0 to $C_{\max}$ whose durations sum to the
makespan — a property the tests check on random instances.

::: linear_opt.problems.jobshop

::: linear_opt.data.jsplib
